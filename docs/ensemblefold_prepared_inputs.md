# EnsembleFold Protenix：MSA、模板与 prepared 输入

本页描述 2026-09-20 wrapper 工作树的新合同。保留 Protenix 的顶层任务列表、
`proteinChain`、`count` 等原生表示；不是把整个输入 schema 改成 AF3。

## 从用户输入到预测

1. 用户给主 JSON，可包含序列、已有 paired/unpaired MSA，以及显式模板。
2. 启用 data 时，缺少 MSA 的蛋白沿原生服务路线搜索；复合物上下文保留。
   某蛋白只要明确给了任一 MSA 通道，就按原生“已提供”处理，不自动补另一通道。
   其他蛋白的搜索也不得覆盖它。坏路径报错，不偷偷退回搜索。
3. data 和 `use_template` 同时开启时，缺省/null 的 `templates` 触发原生模板搜索：
   从当前实际 paired/unpaired 内容构建搜索输入，继续使用原生 Hmmsearch、日期筛选、
   命中排序、去重、结构获取、选链及必要的 Kalign 重比对。
4. 已选中的模板裁成单链 CIF，并在主 JSON 内记录最终 query/template 索引。
   模板完整 polymer sequence、缺失残基位置、实际选中链、概率和日期信息保留；
   不使用“仅有坐标的残基序列”重编号。裁剪后以原生原子读取器核对坐标与掩码。
5. `write_input_json=true` 才持久化 prepared；否则搜索/转换结果只在私有运行空间中
   交给后续消费者。仅 data 且 write=false 的结果不保留，不适合准备供下一次预测的输入。
6. 实际执行 inference 时重新读取当前 JSON 和引用的文件，生成 MSA/模板特征。
   inference-only 不搜索模板、不查询模板结构数据库，也不重新运行模板命中筛选。
   原生 MSA 配对、统计、裁剪与模型特征处理不被新的文件布局替代。

## 模板输入

```json
[
  {
    "name": "job",
    "modelSeeds": [101],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "GHC",
          "count": 1,
          "pairedMsa": "",
          "unpairedMsaPath": "my_deepmsa.a3m",
          "templates": [
            {
              "mmcifPath": "my_single_chain.cif",
              "queryIndices": [0, 1, 2],
              "templateIndices": [0, 1, 2]
            }
          ]
        }
      }
    ]
  }
]
```

- `mmcifPath` 相对**主 JSON 所在目录**解析；也可用 `mmcif` 内联文本，二者只能给一个。
- CIF 必须只有一个蛋白链，且具备完整 polymer sequence 元数据。
  不接受只含坐标、无法确定缺失残基编号的简化 CIF。
- 双索引为零基完整序列位置，等长、非空、一一对应、不得越界。
  例如模板第 5 个残基无坐标，它仍占索引 4；映射到它的坐标掩码为零，后面的残基不前移。
- 不再接受 `chainId` 或 `templatesPath`。用户提供多链结构时，先提取想用的单链；
  wrapper 的自动搜索则会自行从原生实际选中的链导出。
- 显式 CIF 没有索引时不自动对齐。错误的显式模板报错，不静默丢弃。
- 自动模板的正常候选拒绝/无命中仍沿原生路线；日志保留诊断并报告保留数量。
  没有可用模板且有真实处理错误时失败；仅日期未知的既有降级策略保留。

| 输入 | data + use_template=true | inference-only + use_template=true |
| --- | --- | --- |
| 没有 templates / null | 按原生路线搜索；需有可用 MSA | 无模板，不搜索 |
| templates: [] | 明确不使用模板，不搜索 | 无模板 |
| 非空 templates 列表 | 校验并使用用户模板，不搜索 | 读取当前 CIF 和索引生成特征 |

`use_template=false` 是总开关，不使用模板，原生默认值仍是 false。
模型是否支持模板仍由原生模型版本决定。
写 prepared 时仍校验并保存用户提供的资源，不因暂时关闭模板使用而删除模板。

### 不使用持久模板解析缓存

wrapper 的模板准备与推理不支持非空 `prot_template_cache_dir`；若配置了该目录，
进入模板处理时明确报错，要求设为空字符串或 `None`。自定义传入的模板处理器也不能
带持久解析缓存，以免选择使用旧结构、prepared 导出却使用当前 CIF。
默认配置原本为空，因此正常用法不变；inference 仍从当前 JSON/CIF 重建特征。
这里禁用的是预解析结构 `.pkl` 缓存，不是 CIF 文件、发布日期表或 obsolete-PDB 表。
原生训练和独立缓存工具仍保留原有机制，已有缓存文件不会被删除。

## prepared 文件布局与写出

```text
<out_dir>/<job>/
  <job>_data.json
  msas/
    <job>__A_pairedmsa.a3m.zst
    <job>__A_unpairedmsa.a3m.zst
    <job>__A_template_0.cif.zst
    <job>__B_...
  models/                 # 推理结果，原有输出合同不变
  summary_confidences/
  full_data/
```

显式空 MSA 保持 JSON 内空字符串，不强行生成空文件。关闭压缩时使用 `.a3m` / `.cif`。
所有新模板都在主 JSON 的 `templates` 列表中；不生成附属模板 JSON 或公开 HHR 命中列表。
JSON 中的引用形如 `msas/job__A_template_0.cif.zst`。
外部 MSA／模板会先被读取，再复制/转换进 bundle；搬动完整任务目录可保留这些资源引用。
`FILE_` 配体仍可能引用目录外文件，不在上述打包保证内；移动前应另行检查其路径。

`--write_input_json` 默认 true，data、仅推理和全部 skip 时都刷新当前输入快照；可明确 false。
true 时即便原来的 `_data.json` 已存在也会更新；false 时既不生成也不覆盖公开 prepared。
搜索、拆分输入、ESM embedding 等运行中间文件放独立临时目录：
优先有效的 `SLURM_TMPDIR`，其次系统临时目录（尊重 `TMPDIR`）。
正常返回和 Python 捕获的异常都会清理；SIGKILL/节点故障仍需系统或用户清理。
共享权重、CCD 等数据库不属于此临时目录，不会删除。

写出先暂存所有输入资源，再逐任务发布、最后替换该任务 JSON；捕获到发布失败时回滚
该任务已替换文件。允许输出资源也是本次输入，避免交换路径时边读边覆盖。
这是单写者合同，不是并发读写快照，也不保证整个多任务批次一起提交/回滚。
旧 bundle 中本轮不再引用的历史文件不会自动删除，避免误删用户文件。

## 常见实验：只换 unpaired

保留 prepared JSON 的 paired 路径与 templates 列表，只修改
`unpairedMsaPath` 指向 DeepMSA2 A3M，然后用 data=false 运行。
这条路线读取新的 unpaired，不重新搜索、不替换 paired，也不因换 MSA 而重搜模板。
默认 write_input_json=true 将当前条件保存成新的自包含快照；显式 false 才禁止公开写出。

`--compress_fold_input` 默认 false，写外置 `.a3m` / `.cif`；true 写上文的 `.zst`。
`--compress_full_confidence` 默认 false，详细置信度写 JSON，true 写压缩 NPZ；
不会启用默认关闭的详细置信度输出。两种格式切换后清理同一样本的旧格式文件，
输入读取不受写出压缩选项限制。预测仍在每个 seed 完成后立即写出。

注意：原来约定的 skip 仍只看 seed/sample 必需文件存在且非空。
输入改变不会自动使完成的 seed 失效；要真正重算，请换输出目录或关闭 skip。

## 范围与旧示例

本合同适用于 `protenix pred` / `run_protenix.sh` 的统一工作流。
低层 prep/mt 工具仍保留自己的搜索产物/命名；模板内容同样内联，但不代表公开 bundle 写出合同。
原生训练流水线和独立 Web service RequestParser 不是该入口，未在本轮改造。
历史 examples/examples_with_template 和 example_with_json_template 是旧原生模板格式，
不能直接输入新 wrapper。见新的 [单链模板示例](../examples/ensemblefold_inline_template.json)。

本轮验证是离线 CPU 合同、运行生命周期和小型 CIF 原生特征对照；
不宣称已完成在线数据库搜索、真实大复合物、GPU 或全六方法验收。
