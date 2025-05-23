# tuneqwen

#### Description
大模型算法与实践课程作业。
使用nuggets筛选法律问答数据集(DISC-Law-SFT)中的高质量数据并进行SFT微调。

训练完毕的模型链接:(上传中，待添加)

#### Installation

1.  conda env create -f environment.yml
2.  download [DISC-Law-SFT](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)

#### Instructions

1.  split dataset: `python dataset_split.py`
2.  filter high quality data `bash filter_faster.sh`
3.  train the model `bash train_sft.sh`
4.  evlaute the trained model `evaluate_trained.sh`
(Remember to modify the path to that of ours.)

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
