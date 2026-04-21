import matplotlib.pyplot as plt
import numpy as np
import os

def generate_comparison_plots():
    # Ensure images directory exists
    os.makedirs('images', exist_ok=True)

    # 1. Rating Accuracy Plot (MAE & RMSE)
    labels = ['MAE', 'RMSE']
    baseline_scores = [0.8173, 1.0537]
    neural_scores = [0.7989, 1.0603]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Cosine Baseline', color='#ff9999', edgecolor='black')
    rects2 = ax.bar(x + width/2, neural_scores, width, label='Neural MF', color='#66b3ff', edgecolor='black')

    ax.set_ylabel('Error Score (Lower is Better)')
    ax.set_title('Rating Prediction Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.25)

    # Attach labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig('images/accuracy_comparison.png', dpi=300)
    plt.close()
    print("Generated: images/accuracy_comparison.png")

    # 2. Ranking Metrics Plot
    labels = ['Precision@10', 'Recall@10', 'F1-Measure', 'NDCG@10']
    baseline_ranking = [0.0010, 0.0001, 0.0002, 0.0047]
    neural_ranking = [0.0518, 0.0240, 0.0328, 0.1802]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, baseline_ranking, width, label='Cosine Baseline', color='#ffcc99', edgecolor='black')
    rects2 = ax.bar(x + width/2, neural_ranking, width, label='Neural MF', color='#99ff99', edgecolor='black')

    ax.set_ylabel('Score (Higher is Better)')
    ax.set_title('Top-10 Recommendation Ranking Quality')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.55)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig('images/ranking_comparison.png', dpi=300)
    plt.close()
    print("Generated: images/ranking_comparison.png")

if __name__ == '__main__':
    generate_comparison_plots()
