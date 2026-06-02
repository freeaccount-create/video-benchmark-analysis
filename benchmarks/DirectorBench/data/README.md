# Data Directory

## Video Type Definitions

The `video_type` field in metadata must be one of the following canonical values:

| `video_type` (English key) | 中文名称 | Description |
|---|---|---|
| `narrative` | 叙事/故事类 | Story-driven videos with character arcs and plot progression |
| `cinematic` | 电影镜头类 | Director-level cinematic shots emphasizing camera technique |
| `sci_fi` | 科幻类（反规律） | Science fiction / physics-defying / surreal content |
| `action` | 动作类 | Action sequences, fights, chases, stunts |
| `vlog` | 日常生活（vlog） | Daily life, lifestyle, slice-of-life content |
| `commercial` | 商业营销 | Brand promotion, product showcase, advertisements |
| `educational` | 教育新闻 | Educational content, news reporting, documentaries |
| `music` | 音乐类 | Music videos, lyric videos, performance recordings |

## CSV Input Formats

### Format A — Crafted Data
One row per case. Key columns:

| Column | Description |
|---|---|
| `case_id` | Unique case identifier |
| `account_name` | Test account name |
| `title` | Category title |
| `scene` | Scene/category label (e.g. 影视/故事短片, 营销) |
| `prompt` | User's initial video generation prompt |
| `total_turns` | Number of conversation turns |
| `edit_tool_status` | Generation result status |
| `visualization_url` | URL to the visualization result |

### Format B — Agent Online Data
Multi-row conversation format. Key columns:

| Column | Description |
|---|---|
| `用户序号` | User sequence number (only filled on first message of each user) |
| `消息序号` | Message sequence number within conversation |
| `角色` | Role: `user` / `assistant` / `tool` |
| `内容` | Message content |

> **Note**: `用户序号` is only set on the first row of each user's conversation block. Subsequent rows for the same user have an empty `用户序号` — the extractor propagates the last seen value.

## Output Format

Extracted metadata is saved as JSON files in `data/metadata/` (one file per case), following the schema defined in `directorbench/metadata/schema.py` extended with DirectorBench-specific fields.

See `data/samples/sample_metadata.json` for a full example of the base schema.
