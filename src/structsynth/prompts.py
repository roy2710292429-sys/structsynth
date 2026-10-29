from __future__ import annotations

import json
from typing import Any


JSON_RULES = """只输出一个合法 JSON 对象，不要 Markdown 代码围栏，不要解释。
所有字段必须存在；无法判断时使用 null，不要编造输入中不存在的编号。"""


def json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


SEGMENT_SYSTEM = f"""你是题目结构与思维过程分析器。把一个题目切成按出现顺序排列的最小可复用语义语段。
每个语段除了原文，还要给出：
- role：结构角色，如对象、初始条件、约束、中间操作、目标、验证、答案格式；
- operation：该语段要求的认知操作，如提取、比较、转换、推导、综合、验证；
- abstraction：去掉具体数字和题材后的抽象功能。
语段应可在同类题之间复现，不要把整题压成一句，也不要切到短语碎片。
role/operation/abstraction 要尽量使用稳定词汇，便于比较不同的表面表述。
{JSON_RULES}
输出格式：
{{"questions":[{{"question_id":"原编号","segments":[
{{"text":"语段1","role":"初始条件","operation":"提取","abstraction":"给出可用信息"}},
{{"text":"语段2","role":"目标","operation":"推导","abstraction":"求目标量"}}
]}}]}}"""


def segment_user(questions: list[dict[str, str]]) -> str:
    return f"""TASK:SEGMENT
<seed_questions>
{json_block(questions)}
</seed_questions>
请逐题切分。"""


PROFILE_SYSTEM = f"""你是题目思维难度分析器。根据一组同类题，归纳它们共同要求的思维能力，而不是统计题面长度。
特别关注：必要推理步骤数、步骤间依赖、隐含条件、跨概念组合、验证负担、容易卡住的瓶颈。
相似题可以有不同表面场景，但只要核心依赖链和认知操作相近，就属于同一结构族。
{JSON_RULES}
输出格式：
{{"depth_level":3,"depth_label":"中等偏深","abstraction_level":"具体到抽象",
"operations":["条件转译","多步推导","结果验证"],
"dependency_chain":["读取并抽象条件","选择关系/方法","执行关键推导","验证目标是否满足"],
"bottleneck":"最需要洞察或跨概念连接的一步",
"verification_burden":"验证要求",
"surface_variation":["可以改变的具体维度1","可以改变的具体维度2"],
"depth_invariants":["必须保持的思维约束1","必须保持的思维约束2"]}}"""


def profile_user(seeds: list[dict[str, Any]]) -> str:
    return f"""TASK:INDUCE_REASONING_PROFILE
<seed_questions>
{json_block(seeds)}
</seed_questions>
请归纳共同的思维深度画像。"""


SLOT_SYSTEM = f"""你是“同类题模板”归纳器。输入是若干候选语义单元，每个单元来自多个种子题中的语段。
任务：
1. 合并同一个结构角色的重复候选，删除只表达具体题材、数字或专有名词的候选。
2. 给每个保留单元一个简洁的中文槽位名和可操作定义。
3. 仅依据语段的典型出现顺序确定槽位顺序；相同位置可并列。
4. 写出跨题保持不变的结构不变量，不要写某一个种子题的具体内容。
5. 槽位是软模板：同类题可以省略非关键连接词、换场景或改变数字，但依赖关系和认知操作应保持。
{JSON_RULES}
输出格式：
{{"name":"模板名","description":"模板定义",
"slots":[{{"source_unit_ids":["候选编号1","候选编号2"],"name":"槽位名",
"definition":"该槽放什么",
"required":true,"order":1}}],
"invariants":["不变量1","不变量2"]}}"""


def slot_user(
    candidates: list[dict[str, Any]],
    reasoning_profile: dict[str, Any] | None = None,
) -> str:
    return f"""TASK:INDUCE_TEMPLATE
<candidates>
{json_block(candidates)}
</candidates>
<reasoning_profile>
{json_block(reasoning_profile or {})}
</reasoning_profile>"""


def generation_system(arm: str) -> str:
    if arm == "baseline":
        condition = """条件是：仅阅读种子题文本。自行推断共同题目类型并仿写。
不要解释，不要直接改写原题；改变题面对象、数字和具体叙事，同时保持所属题型。
重点保持思维步骤数量、步骤依赖和主要难点，不要用更浅的套公式题替换原题。"""
    elif arm == "matrix":
        condition = """条件是：种子题，以及直接从语段表示 M 聚类得到、但未使用谱分解的无序语义单元。
把候选单元当作软提示；允许调整顺序和表面表达。
重点保持思维步骤数量、依赖关系和认知操作，不要为了“像”而只复用题材或句式。"""
    elif arm == "spectral":
        condition = """条件是：种子题，以及由 A=2 sym(M)-sym(N) 的谱结构聚类得到的模板槽位。
必须表达给定槽位承载的思维依赖链，但不要求逐字对应槽位或沿用原题材。
优先满足 depth_invariants、dependency_chain 和 bottleneck，再考虑槽位名称。
允许合并同义连接语、替换场景、改变数字和组合方式；不要生成只是换了名字的浅层仿写。"""
    else:
        raise ValueError(f"Unknown generation arm: {arm}")

    return f"""你是严格受控的题目生成器。{condition}
{JSON_RULES}
输出格式：
{{"questions":[{{"text":"完整新题","slot_texts":{{"槽位编号":"该槽在本题中的文本"}},
"type":"题目类型","difficulty":"easy|medium|hard","depth_level":1,
"reasoning_outline":["抽象思维步骤1","抽象思维步骤2"]}}]}}"""


def generation_user(
    arm: str,
    seeds: list[dict[str, Any]],
    count: int,
    template: dict[str, Any] | None,
    candidates: list[dict[str, Any]] | None,
) -> str:
    condition = {}
    if arm in {"matrix", "spectral"}:
        condition["candidate_units"] = candidates or []
    if template is not None:
        condition["template"] = template
    return f"""TASK:GENERATE
<seed_questions>
{json_block(seeds)}
</seed_questions>
<condition>
{json_block(condition)}
</condition>
请生成 {count} 道彼此不同的新题。arm 参数（仅用于日志，不得写入题目）：{arm}"""


JUDGE_SYSTEM = f"""你是盲评结构判定器。你只看到参考模板和若干生成题，不知道生成条件属于哪个实验组。
对每题判断它是否属于参考模板的同一结构族，并估计思维深度是否对齐。
不要按题材、术语、句式或具体数值的相似度打分。允许灵活换场景、合并同义语段和更换表达方式。
structural_similarity 评估依赖链、认知操作和目标约束的连续相似程度；
same_family 表示是否仍可归入同一题目结构族；
cognitive_coverage 表示关键认知操作被覆盖的比例；
depth_score 为 1-5；depth_alignment 为 0-1，衡量其深度、步骤数和瓶颈是否与画像相称。
coverage 必须是命中的 required 槽位比例，不是文字相似度。
若题目没有提供 slot_texts，必须从完整题面推断槽位是否被表达，不能因此判为零。
{JSON_RULES}
输出格式：
{{"results":[{{"question_id":"原编号","isomorphic":true,"same_family":true,
"structural_similarity":0.0,"cognitive_coverage":0.0,
"depth_score":1,"depth_alignment":0.0,
"matched_slot_ids":["槽位编号"],"missing_slot_ids":["槽位编号"],
"distorted_slot_ids":["槽位编号"],"coverage":0.0,"reason":"一句证据"}}]}}"""


def judge_user(template: dict[str, Any], questions: list[dict[str, Any]]) -> str:
    return f"""TASK:JUDGE
<reference_template>
{json_block(template)}
</reference_template>
<generated_questions>
{json_block(questions)}
</generated_questions>"""
