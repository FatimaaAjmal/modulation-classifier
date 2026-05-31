# Automatic Modulation Classification using Deep Learning

## Overview
This project implements a deep learning model to automatically classify 
radio signal modulation types - BPSK, QPSK, and 16-QAM, using a neural 
network built in PyTorch.

## Research Background
This project extends my IEEE-published research on Software Defined Radio 
(SDR) and QAM modulation schemes. My 2009 publications investigated optimal 
QAM constellation design and SDR physical layer security. This project asks 
the next natural question: can a neural network automatically identify these 
modulation types from raw signal data?

This is an active research problem in 6G intelligent radio systems, 
Integrated Sensing and Communications (ISAC), and cognitive radio — where 
radios must sense and adapt to their environment automatically.

## What it Does
- Generates synthetic BPSK, QPSK and 16-QAM signals with realistic AWGN noise
- Trains a fully connected neural network to classify modulation type
- Achieves ~77% classification accuracy on test data
- Visualises training loss and classification accuracy

## Files
| File | Description |
|------|-------------|
| `generate_signals.py` | Generates synthetic radio signals with noise |
| `train_model.py` | Builds, trains and evaluates the neural network |
| `results.png` | Training loss and accuracy graphs |

## Results
The model achieves approximately 77% classification accuracy, compared to 
33% random baseline for a 3-class problem, demonstrating that the neural 
network successfully learns to distinguish between modulation types.

![Results](results.png)

## Technologies
- Python 3.14
- PyTorch
- NumPy
- Matplotlib

## Related IEEE Publications
- *Secure End-to-End Communication over GSM and PSTN Networks* — 22 citations
- *Developing and Implementing Encryption Algorithm for GSM Security Issues* - 14 citations
- *Effective Normal Binary Ordering in QAM for Enhanced Voice Quality in SDR*
- *Enhancing SDR Security Using Gray Coded Mapping Scheme in Rectangular QAM*

**Best Paper Award** - 2nd IEEE International Conference on Computer, 
Control and Communication, Karachi, 2009

## Author
**Fatima Ajmal**
B.E. Telecommunications Engineering, NUST Pakistan (CGPA 3.68)
[Google Scholar](https://scholar.google.com) | fatima_ajmal@yahoo.com
