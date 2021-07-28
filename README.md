# Demonstration-Guided Reinforcement Learning with Learned Skills








## Datasets

|Dataset        | Link         | Size |
|:------------- |:-------------|:-----|
| Maze Task-Agnostic | [https://drive.google.com/file/d/103RFpEg4ATnH06fd1ps8ZQL4sTtifrvX/view?usp=sharing](https://drive.google.com/file/d/103RFpEg4ATnH06fd1ps8ZQL4sTtifrvX/view?usp=sharing)| 470MB |
| Maze Demos | [https://drive.google.com/file/d/1wTR9ns5QsEJnrMJRXFEJWCMk-d1s4S9t/view?usp=sharing](https://drive.google.com/file/d/1wTR9ns5QsEJnrMJRXFEJWCMk-d1s4S9t/view?usp=sharing)| 100MB |
| Office Cleanup Task-Agnostic | [https://drive.google.com/file/d/1FOE1kiU71nB-3KCDuxGqlAqRQbKmSk80/view?usp=sharing](https://drive.google.com/file/d/1FOE1kiU71nB-3KCDuxGqlAqRQbKmSk80/view?usp=sharing)| 170MB |
| Office Cleanup Demos | [https://drive.google.com/file/d/149trMTyh3A2KnbUOXwt6Lc3ba-1T9SXj/view?usp=sharing](https://drive.google.com/file/d/149trMTyh3A2KnbUOXwt6Lc3ba-1T9SXj/view?usp=sharing)| 6MB |

To download the dataset files from Google Drive via the command line, you can use the 
[gdown](https://github.com/wkentaro/gdown) package. Install it with:
```
pip install gdown
```

Then navigate to the folder you want to download the data to and run the following commands:
```
# Download Maze Task-Agnostic Dataset
gdown https://drive.google.com/uc?id=103RFpEg4ATnH06fd1ps8ZQL4sTtifrvX

# Download Maze Demonstration Dataset
gdown https://drive.google.com/uc?id=1wTR9ns5QsEJnrMJRXFEJWCMk-d1s4S9t

# Download Office Task-Agnostic Dataset
gdown https://drive.google.com/uc?id=1FOE1kiU71nB-3KCDuxGqlAqRQbKmSk80

# Download Office Demonstration Dataset
gdown https://drive.google.com/uc?id=149trMTyh3A2KnbUOXwt6Lc3ba-1T9SXj
``` 