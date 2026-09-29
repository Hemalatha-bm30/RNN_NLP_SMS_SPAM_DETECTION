# SVL Technologies - NLP + Regex + RNN + FastAPI Project

## Project

Smart SMS Spam Detection using NLP, Regex, and PyTorch RNN.

This project classifies SMS messages as either **Ham** or **Spam** using Natural Language Processing and a PyTorch Recurrent Neural Network (RNN). The trained model is integrated with FastAPI for real-time prediction through Swagger UI.

## Flow

Raw SMS → Regex Cleaning → Tokenization → Vocabulary → Sequence Padding → PyTorch RNN → Spam/Ham → FastAPI → Swagger UI

## Project Structure

```text
RNN_NLP_SMS_Project/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── sms_messages.csv
│
├── models/
│   └── sms_rnn.pt
│
├── model.py
├── preprocess.py
├── train.py
├── requirements.txt
└── README.md