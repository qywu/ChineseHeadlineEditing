# Importance-Aware Learning for Neural Headline Editing



## Requirements

The two packackages below need to install manually.

* For TorchFly, you need to install [apex](https://github.com/qywu/apex) first.

    ```bash
    # modified the error due to cuda version
    git clone https://github.com/qywu/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```

* You need to install OpenCC (https://github.com/BYVoid/OpenCC) for text pre-processing.

    ```bash
    sudo apt-get install opencc
    pip install opencc-python-reimplemented
    ```

For the rest packages, use:

```bash
pip install requirements.txt
```

## PHED dataset and Headline Generation dataset

You can download our processed version and extract it under `/data`. [[Google Drive]](https://drive.google.com/file/d/1D2WluG_3LLKM81NyZLvUgtzUFlJrgfE0/view?usp=sharing)

Or you can download [PHED](https://drive.google.com/file/d/1UOtCsk9JfO-lSdV9jaeIc6tj8QdC07y8/view?usp=sharing) and [Headline Generation](https://drive.google.com/file/d/1me9-JHapu6DlvXL7OEET7-tsGRoLkCMU/view?usp=sharing) datasets separately and process them with code under `/notebooks`.

## Training

See `/notebooks/Train PAS.ipynb`

## Released Weights

[Final PAS weights](https://drive.google.com/file/d/1WOnV6xMAvokd-aOBTehxHeojz91OiJkl/view?usp=sharing)

[After Adaptation weights](https://drive.google.com/file/d/1nG_V1tVTNgabLo95yy7PIBH-4zpo7ji1/view?usp=sharing)

## Chinese GPT

[Chinese GPT weights](https://drive.google.com/file/d/1agi64d06PlBe6XUz2IMkgl8ZKjw6H7nS/view?usp=sharing)

We will later try to upload to huggingface's custom model repo. For now, you can try google colab example: https://colab.research.google.com/drive/1cvBSt2uF7hYL1feDGt0dkCxIeaVXQs5x

