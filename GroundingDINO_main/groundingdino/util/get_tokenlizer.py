from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
import os

def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))

    #tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    # ====================== 【关键修改：强制本地路径+local_files_only】 ======================
    # 原代码：tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    local_bert_path = "D:/groundingdino_work/bert-base-uncased"  # 写死本地路径
    tokenizer = AutoTokenizer.from_pretrained(local_bert_path, local_files_only=True)  # 加local_files_only
    # ====================================================================================

    return tokenizer


# def get_pretrained_language_model(text_encoder_type):
#     if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
#         return BertModel.from_pretrained(text_encoder_type)
#     if text_encoder_type == "roberta-base":
#         return RobertaModel.from_pretrained(text_encoder_type)

#     raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))


def get_pretrained_language_model(text_encoder_type):
    # ====================== 【同步修改：get_pretrained_language_model也加local_files_only】 ======================
    # 原代码：return BertModel.from_pretrained(text_encoder_type)
    local_bert_path = "D:/groundingdino_work/bert-base-uncased"
    if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        return BertModel.from_pretrained(local_bert_path, local_files_only=True)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type, local_files_only=True)
    # ============================================================================================================
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
