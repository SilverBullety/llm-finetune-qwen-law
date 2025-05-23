# tuneqwen

#### 介绍
大模型算法与实践课程作业。
使用nuggets筛选法律问答数据集(DISC-Law-SFT)中的高质量数据并进行SFT微调。

训练完毕的模型链接:
通过网盘分享的文件：weights.zip
链接: https://pan.baidu.com/s/13RUzhovR2UXZFFNMqxoD1A?pwd=n5se 提取码: n5se 
--来自百度网盘超级会员v1的分享
(刚发现这一版的weights好像忘记保存tokenizer了，正在重新上传一版... 
用这一版的话可能得手动下一下qwen-7b的tokenizer:https://huggingface.co/Qwen/Qwen-7B, 下载tokenizer_config.json、tokenization_qwen.py、qwen.tiktoken即可使用)

#### 安装教程

1.  conda env create -f environment.yml
2.  download [DISC-Law-SFT](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)

#### 使用说明

1.  划分数据集: `python dataset_split.py`
2.  筛选高质量数据 `bash filter_faster.sh`
3.  训练模型 `bash train_sft.sh`
4.  评估训练后的模型 `evaluate_trained.sh`
(请记得将脚本中的路径改为你自己的路径)

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request
