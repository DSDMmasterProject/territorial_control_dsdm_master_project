import pandas as pd
df = pd.read_csv('model/feature_matrix.csv')
changed = df[df['changed'] == 1]
print("Among cells that changed:")
print(f"  With any events (this month): {(changed['n_events']>0).mean()*100:.1f}%")
print(f"  With any lag1 events: {(changed['n_events_lag1']>0).mean()*100:.1f}%")
print(f"  With any lag3 events: {(changed['n_events_lag3']>0).mean()*100:.1f}%")
print(f"  Mean events: {changed['n_events'].mean():.3f}")
print(f"  Mean neighbor events: {changed['neighbor_n_events'].mean():.3f}")