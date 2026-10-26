import AppKit
import CoreText
import Foundation
import ImageIO
import PDFKit
import Vision

/// Apple Vision OCR helper for local-doc-kb.
///
///   visionocr IMAGE
///   visionocr --pdf-max N --pdf-dpi DPI FILE.pdf
///   visionocr --pdf-page N --pdf-dpi DPI FILE.pdf
///   visionocr --self-test
///
/// Multi-page PDF stdout is parsed by build_ocr.py:
///   ===== page N chars=C sec=S =====
///   <text>

struct Flags {
    var pdfMax: Int?
    var pdfPage: Int?
    var dpi: CGFloat = 150
    var selfTest = false
    var path: String?
}

func fail(_ message: String) -> Never {
    fputs("visionocr: \(message)\n", stderr)
    exit(1)
}

func parseArgs() -> Flags {
    var f = Flags()
    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0
    while i < args.count {
        let a = args[i]
        func needValue() -> String {
            i += 1
            if i >= args.count { fail("missing value for \(a)") }
            return args[i]
        }
        switch a {
        case "--pdf-max":
            guard let n = Int(needValue()), n > 0 else { fail("bad --pdf-max") }
            f.pdfMax = n
        case "--pdf-dpi":
            guard let n = Double(needValue()), n > 0 else { fail("bad --pdf-dpi") }
            f.dpi = CGFloat(n)
        case "--pdf-page":
            guard let n = Int(needValue()), n > 0 else { fail("bad --pdf-page") }
            f.pdfPage = n
        case "--self-test":
            f.selfTest = true
        case _ where a.hasPrefix("-"):
            fail("unknown argument \(a)")
        default:
            if f.path != nil { fail("one input file only") }
            f.path = a
        }
        i += 1
    }
    return f
}

func recognize(_ image: CGImage) throws -> String {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    if #available(macOS 13.0, *) {
        request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]
    }
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try handler.perform([request])
    let lines = (request.results ?? []).sorted { a, b in
        if abs(a.boundingBox.midY - b.boundingBox.midY) > 0.01 {
            return a.boundingBox.midY > b.boundingBox.midY
        }
        return a.boundingBox.minX < b.boundingBox.minX
    }
    return lines.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
}

func loadImage(path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let src = CGImageSourceCreateWithURL(url, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func render(page: PDFPage, dpi: CGFloat) -> CGImage? {
    let bounds = page.bounds(for: .mediaBox)
    let scale = dpi / 72.0
    let w = Int((bounds.width * scale).rounded(.up))
    let h = Int((bounds.height * scale).rounded(.up))
    guard w > 0, h > 0 else { return nil }
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    ctx.setFillColor(CGColor(gray: 1, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    ctx.saveGState()
    ctx.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: ctx)
    ctx.restoreGState()
    return ctx.makeImage()
}

func pdfPages(path: String, dpi: CGFloat, only: Int?, maxPages: Int?) -> [(Int, String, Double)] {
    let url = URL(fileURLWithPath: path)
    guard let doc = PDFDocument(url: url) else { fail("cannot open pdf \(path)") }
    let count = doc.pageCount
    if count == 0 { return [] }
    var indexes: [Int] = []
    if let only {
        if only > count { fail("pdf has \(count) pages, asked for \(only)") }
        indexes = [only - 1]
    } else {
        let cap = min(count, maxPages ?? count)
        indexes = Array(0..<cap)
    }
    var out: [(Int, String, Double)] = []
    for idx in indexes {
        let t0 = Date()
        guard let page = doc.page(at: idx), let image = render(page: page, dpi: dpi) else {
            out.append((idx + 1, "", Date().timeIntervalSince(t0)))
            continue
        }
        let text = (try? recognize(image)) ?? ""
        out.append((idx + 1, text, Date().timeIntervalSince(t0)))
    }
    return out
}

func emitPages(_ pages: [(Int, String, Double)], single: Bool) {
    if single, let first = pages.first {
        print(first.1)
        return
    }
    for (n, text, sec) in pages {
        print("===== page \(n) chars=\(text.count) sec=\(String(format: "%.2f", sec)) =====")
        if !text.isEmpty { print(text) }
    }
}

func writeCanaryPDF(_ url: URL, _ text: String) -> Bool {
    var box = CGRect(x: 0, y: 0, width: 640, height: 280)
    guard let ctx = CGContext(url as CFURL, mediaBox: &box, nil) else { return false }
    ctx.beginPDFPage(nil)
    ctx.setFillColor(CGColor(gray: 1, alpha: 1))
    ctx.fill(box)
    let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 36, nil)
    let attrs: [CFString: Any] = [
        kCTFontAttributeName: font,
        kCTForegroundColorAttributeName: CGColor(gray: 0, alpha: 1),
    ]
    let astr = CFAttributedStringCreate(nil, text as CFString, attrs as CFDictionary)!
    let line = CTLineCreateWithAttributedString(astr)
    ctx.textMatrix = CGAffineTransform(translationX: 36, y: 120)
    CTLineDraw(line, ctx)
    ctx.endPDFPage()
    ctx.closePDF()
    return true
}

func selfTest() -> Int32 {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("local-doc-kb-visionocr-canary.pdf")
    let token = "CANARY 4412"
    guard writeCanaryPDF(url, token) else { fail("cannot write canary pdf") }
    defer { try? FileManager.default.removeItem(at: url) }
    let pages = pdfPages(path: url.path, dpi: 200, only: 1, maxPages: nil)
    let text = pages.first?.1 ?? ""
    if text.contains("CANARY") && text.contains("4412") {
        print("PASS \(token)")
        return 0
    }
    fputs("FAIL vision self-test, got: \(text)\n", stderr)
    return 1
}

let flags = parseArgs()
if flags.selfTest {
    exit(selfTest())
}
guard let path = flags.path else { fail("usage: visionocr [--pdf-max N | --pdf-page N] [--pdf-dpi D] FILE") }
if !FileManager.default.fileExists(atPath: path) { fail("no such file \(path)") }

let isPDF = path.lowercased().hasSuffix(".pdf") || flags.pdfMax != nil || flags.pdfPage != nil
if isPDF {
    let pages = pdfPages(path: path, dpi: flags.dpi, only: flags.pdfPage, maxPages: flags.pdfMax)
    emitPages(pages, single: flags.pdfPage != nil)
} else {
    guard let image = loadImage(path: path) else { fail("cannot read image \(path)") }
    let text = (try? recognize(image)) ?? ""
    if !text.isEmpty { print(text) }
}
exit(0)
