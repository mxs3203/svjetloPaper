
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import NMF
import numpy as np

data = pd.read_csv("data/forNMF_350_850_normalized_by_apogee.csv")
data.drop(columns=data.columns[0], axis=1,  inplace=True)
data = data[data['Meter'] > 1]

curve_columns = data.columns[0:150]
data['meanCurveValue'] = data[curve_columns].mean(axis=1)
data = data[data['meanCurveValue'] < 40] # TODO


tmp = data[data['Cast'] == 3]
tmp = tmp[tmp['Sensor'] == 'DW']
tmp = tmp.melt(id_vars=['Meter'], value_vars=curve_columns)

plt.figure(figsize=(25,12))
p = sb.scatterplot(tmp, y='value', x='variable',
               hue='Meter',
               hue_norm=(0,60),
               palette=sb.color_palette("coolwarm", as_cmap=True))
sb.set(font_scale=2)
p.set_xlabel("nm",fontsize=30)
p.set_ylabel("% Apogee",fontsize=30)
plt.xticks(rotation=90)
plt.title("Cast 7")
plt.show()



res = []
for seed in [1,523,64,66,22,661,55,7,77,22,53,643,777,521,0,27]:
    for c in [2,3,4,5,6,7,8,9,10]:
        for init in ['nndsvd', 'nndsvdar']:
            for s in ['mu', 'cd']:
                for l in ['frobenius','kullback-leibler']:
                    if s == 'cd' and l == 'kullback-leibler':
                        print("cd and kullback solver cannot go together")
                    else:
                        model = NMF(n_components=c, init=init,
                                    random_state=seed, solver=s,
                                    beta_loss=l, max_iter=1000)
                        W = model.fit_transform(data[curve_columns])
                        H = model.components_
                        res.append(pd.DataFrame([c,init,s,l,model.reconstruction_err_]).T)

res = pd.concat(res)
res.columns = ['K', 'init', 'solver', 'distance', 'error']
res_f = res[res['distance'] == 'frobenius']
ax = sb.boxplot(res, x='K', y = 'error', hue='init', boxprops={'alpha': 0.4})
sb.stripplot(data=res, x="K", y="error",
              hue="init", dodge=True, ax=ax)
plt.show()

res_kl = res[res['distance'] == 'kullback-leibler']
ax = sb.boxplot(res, x='K', y = 'error', hue='init', boxprops={'alpha': 0.4})
sb.stripplot(data=res, x="K", y="error",
              hue="init", dodge=True, ax=ax)
plt.show()


model = NMF(n_components=5, init='nndsvdar',
                                random_state=7, solver='mu',
                                beta_loss='kullback-leibler', max_iter=5000)
W = model.fit_transform(data[curve_columns])
H = model.components_
print(c,init,s, l)
print(model.reconstruction_err_)

tmp = pd.DataFrame(H).T
tmp['nm'] = curve_columns.values
tmp = tmp.melt(id_vars=['nm'])
plt.figure(figsize=(25,12))
p = sb.lineplot(tmp, y='value', x='nm',
               hue='variable',
               palette=sb.color_palette("tab10"))
sb.set(font_scale=2)
p.set_xlabel("nm",fontsize=30)
p.set_ylabel("% Apogee",fontsize=30)
plt.xticks(rotation=90)
plt.title("Cast 6")
plt.show()