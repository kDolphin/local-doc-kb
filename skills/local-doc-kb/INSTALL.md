# Installing the local-doc-kb skill into an agent

The skill is just `SKILL.md` (Markdown with YAML front-matter: `name`, `description`).
Product-specific skill paths change over time — check your agent's own docs for
where it loads skills/instructions from. Generic options:

1. **Skills directory** (agents that support `SKILL.md` folders): copy or symlink the
   folder: `ln -s "$PWD/skills/local-doc-kb" <agent-skills-dir>/local-doc-kb`.
   Project-level skills dirs (inside a repo) and user-level dirs (in your home) are
   both common; prefer project-level if the agent works inside this repo.
2. **Project instructions file** (e.g. an `AGENTS.md`-style or rules file the agent
   reads): paste the body of `SKILL.md`, replacing `<repo>` with the absolute path.
3. **Plain system prompt / custom instructions**: paste the "How to query" and
   "Answering rules" sections.

After installing: replace `<repo>` with the real path, start a new agent session,
ask a question answerable from the corpus, and verify the answer cites real relpaths.
The agent needs shell access to run `scripts/ask_hybrid_v2.py` beside the index
it should search.

Grok's default is the cloud computer: install this skill there, copy
`index/*.sqlite3` with `scripts/sync_mirror.sh` after each build, and answer
from that copy. Ask once for the cloud path if config `hosts` is empty. Do not
hardcode a machine path.

Any other agent usually runs on the build Mac. Copy the index only when the
user wants that agent on another computer. The query script and the index
format stay the same. The skill directory and the repo path are what change.
