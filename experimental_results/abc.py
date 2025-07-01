import pandas as pd
import os

# Define whether higher value is better for each metric
metric_direction = {
    'accuracy': True,
    'precision': True,
    'recall': True,
    'f1': True,
    'roc_auc': True,
    'log_loss': False
}

# Decide winner using emoji ranks
def decide_winner(topogat_val, gat_val, metric):
    try:
        t_val = float(topogat_val.split('±')[0].strip())
        g_val = float(gat_val.split('±')[0].strip())

        if abs(t_val - g_val) < 1e-4:
            return "TopoGAT 🥇 / GAT 🥇", (1, 1)
        if metric_direction[metric]:
            return ("TopoGAT 🥇 / GAT 🥈", (1, 0)) if t_val > g_val else ("TopoGAT 🥈 / GAT 🥇", (0, 1))
        else:
            return ("TopoGAT 🥇 / GAT 🥈", (1, 0)) if t_val < g_val else ("TopoGAT 🥈 / GAT 🥇", (0, 1))
    except:
        return "N/A", (0, 0)

# Generate ranking report from a dataset folder
def prepare_report_with_ranking(dataset_path):
    variants = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    rows = []

    for variant in variants:
        variant_path = os.path.join(dataset_path, variant)
        topogat_csv = os.path.join(variant_path, 'summary_topogat.csv')
        gat_csv = os.path.join(variant_path, 'summary_gat.csv')

        if os.path.exists(topogat_csv) and os.path.exists(gat_csv):
            topo_df = pd.read_csv(topogat_csv, index_col=0).T
            gat_df = pd.read_csv(gat_csv, index_col=0).T

            metrics = list(metric_direction.keys())
            topo_row = topo_df.iloc[0]
            gat_row = gat_df.iloc[0]

            row = {'Variant': variant}
            topo_score = 0
            gat_score = 0

            for metric in metrics:
                winner, (t_score, g_score) = decide_winner(topo_row[metric], gat_row[metric], metric)
                row[metric] = winner
                topo_score += t_score
                gat_score += g_score

            if topo_score > gat_score:
                row['Verdict'] = f'TopoGAT Wins 🏆 ({topo_score}🥇)'
            elif gat_score > topo_score:
                row['Verdict'] = f'GAT Wins 🏆 ({gat_score}🥇)'
            else:
                row['Verdict'] = 'Tie 🤝'

            rows.append(row)

    return pd.DataFrame(rows)

# Example usage
dataset_path = '/content/drive/MyDrive/topogat/PTC_MR'
report = prepare_report_with_ranking(dataset_path)
report.to_csv('/content/drive/MyDrive/topogat/PTC_MR/PTC_MR_variant_comparison_report.csv', index=False)
report
