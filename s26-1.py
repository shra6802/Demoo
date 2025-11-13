import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transactions = []
with open("C:/Users/rosha/OneDrive/Desktop/ML Practical/s26-1/groceries.csv") as f:
    for line in f:
        
        transaction = [item.strip() for item in line.strip().split(',')]
        transactions.append(transaction)

print("Sample Transactions:")
for transaction in transactions[:5]:  
    print(transaction)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_onehot, min_support=0.25, use_colnames=True)

print(line)
print("Frequent Itemsets with support >= 0.25:")
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
