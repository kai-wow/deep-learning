# hw1
## 待学习函数
```
nn.LeakyReLU(0.2)
nn.BatchNorm1d(64),
nn.Dropout(0.2)


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
selector = SelectKBest(score_func=f_regression, k=24)
result = selector.fit(raw_x_train, y_train)


optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                weight_decay=1e-3)
# 优化学习率
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                    T_0=2, T_mult=2, eta_min=config['learning_rate'])
scheduler.step()
```

## 不确定的细节
选择特征时， 用拆分完验证集的数据 fit ，还是用所有训练数据 fit
