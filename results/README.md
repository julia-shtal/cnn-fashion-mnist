# Experiment Results

This directory contains saved results from CNN training experiments on the Fashion MNIST dataset.

## 📁 Directory Structure

Each training run creates a **timestamped subdirectory** containing all outputs from that experiment:

```
results/
├── 20251126_143025/                    # Experiment run on Nov 26, 2025 at 14:30:25
│   ├── confusion_matrix_scenario1_uncorrected-to-clean_20251126_143025.png
│   ├── confusion_matrix_scenario2_uncorrected-to-uncorrected_20251126_143025.png
│   ├── confusion_matrix_scenario3_corrected-to-clean_20251126_143025.png
│   ├── confusion_matrix_scenario4_corrected-to-corrected_20251126_143025.png
│   └── experiment_summary_20251126_143025.txt
├── 20251126_150132/                    # Another experiment run
│   └── ...
└── README.md                           # This file
```

---

## 📊 File Types

### Confusion Matrices (PNG)

**Naming Convention:**
```
confusion_matrix_scenario{N}_{description}_{timestamp}.png
```

**Components:**
- `scenario{N}` - Scenario number (1-4)
- `{description}` - Descriptive name of experiment type
- `{timestamp}` - Date and time (YYYYMMDD_HHMMSS)

**Examples:**
- `confusion_matrix_scenario1_uncorrected-to-clean_20251126_143025.png`
- `confusion_matrix_scenario3_corrected-to-clean_20251126_150132.png`

**Visual Format:**
- Size: 10×10 inches (1500×1500 pixels at 150 DPI)
- Color map: Blues (matplotlib)
- Annotations: Cell values with white text on dark cells, black text on light cells
- Labels: True labels on Y-axis, Predicted labels on X-axis

### Experiment Summary (TXT)

**Naming Convention:**
```
experiment_summary_{timestamp}.txt
```

**Contents:**
- Timestamp of experiment
- Error rates for all 4 scenarios
- Scenario descriptions and purposes
- Formatted for easy reading and comparison

**Example:**
```
======================================================================
CNN Fashion MNIST Experiment Results
======================================================================
Timestamp: 20251126_143025
======================================================================

Summary of Results:
----------------------------------------------------------------------
Scenario 1 (Uncorrected→Clean): 0.0987
Scenario 2 (Uncorrected→Uncorrected): 0.0823
Scenario 3 (Corrected→Clean): 0.0891
Scenario 4 (Corrected→Corrected): 0.0645
...
```

---

## 🔬 Experimental Scenarios

### Scenario 1: Uncorrected → Clean
**Training:** Original unbalanced data  
**Testing:** External clean test set  
**Purpose:** Baseline performance on unseen clean distribution

### Scenario 2: Uncorrected → Uncorrected
**Training:** Original unbalanced data  
**Testing:** Split from same unbalanced data  
**Purpose:** Evaluate performance with class imbalance

### Scenario 3: Corrected → Clean
**Training:** Balanced data with oversampling  
**Testing:** External clean test set  
**Purpose:** Impact of balancing on generalization

### Scenario 4: Corrected → Corrected
**Training:** Balanced data with oversampling  
**Testing:** Split from same balanced data  
**Purpose:** Best-case scenario with balanced training and testing

---

## 📈 How to Use These Results

### Viewing Confusion Matrices

**Option 1: Image Viewer**
```bash
# Windows
start results/20251126_143025/confusion_matrix_scenario1_*.png

# Mac
open results/20251126_143025/confusion_matrix_scenario1_*.png

# Linux
xdg-open results/20251126_143025/confusion_matrix_scenario1_*.png
```

**Option 2: Python**
```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('results/20251126_143025/confusion_matrix_scenario1_uncorrected-to-clean_20251126_143025.png')
plt.imshow(img)
plt.axis('off')
plt.show()
```

### Reading Summary Files

```bash
# View in terminal
cat results/20251126_143025/experiment_summary_20251126_143025.txt

# Or open in text editor
notepad results/20251126_143025/experiment_summary_20251126_143025.txt  # Windows
nano results/20251126_143025/experiment_summary_20251126_143025.txt     # Linux
```

### Comparing Multiple Experiments

```python
import os
import re

# List all experiment directories
results_dirs = sorted([d for d in os.listdir('results') if os.path.isdir(os.path.join('results', d))])

print("All Experiments:")
for exp_dir in results_dirs:
    summary_file = f'results/{exp_dir}/experiment_summary_{exp_dir}.txt'
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            content = f.read()
            # Extract error rates
            scenario1 = re.search(r'Scenario 1.*?: (\d+\.\d+)', content)
            if scenario1:
                print(f"{exp_dir}: Scenario 1 Error = {scenario1.group(1)}")
```

---

## 📊 Analyzing Results

### Understanding Confusion Matrices

**Reading the Matrix:**
- **Rows**: True labels (actual class)
- **Columns**: Predicted labels (model's prediction)
- **Diagonal**: Correct predictions (darker = more correct)
- **Off-diagonal**: Misclassifications

**Example Interpretation:**
```
          Predicted
          0   1   2   ...
True  0  [450] 3   5   ...   ← Class 0: 450 correct, 3 misclassified as 1, etc.
      1   2  [480] 1   ...   ← Class 1: 480 correct, 2 misclassified as 0, etc.
      2   5   1  [470] ...   ← Class 2: 470 correct, 5 misclassified as 0, etc.
```

**Common Patterns:**
- **Strong diagonal**: Good overall performance
- **Confusion between similar classes**: Expected (e.g., Shirt vs T-shirt)
- **Weak performance on specific classes**: May indicate class imbalance or difficult samples

### Comparing Scenarios

**Key Questions:**
1. **Does balancing help?** Compare Scenario 1 vs Scenario 3
2. **Overfitting check:** Compare Scenario 2 vs Scenario 1
3. **Best achievable performance:** Check Scenario 4
4. **Generalization gap:** Compare within-distribution (2,4) vs out-of-distribution (1,3) performance

---

## 🛠️ Advanced Analysis

### Extract All Error Rates

```python
import os
import glob
import re
import pandas as pd

def extract_results():
    results = []

    for exp_dir in glob.glob('results/*/'):
        timestamp = os.path.basename(exp_dir.rstrip('/'))
        summary_file = f'{exp_dir}experiment_summary_{timestamp}.txt'

        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                content = f.read()

            row = {'timestamp': timestamp}
            for i in range(1, 5):
                match = re.search(f'Scenario {i}.*?: (\d+\.\d+)', content)
                if match:
                    row[f'scenario_{i}'] = float(match.group(1))

            results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values('timestamp')
    return df

# Usage
df = extract_results()
print(df)

# Save to CSV
df.to_csv('results/all_experiments.csv', index=False)
print("Saved to results/all_experiments.csv")
```

### Visualize Performance Over Time

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/all_experiments.csv')

plt.figure(figsize=(12, 6))
for i in range(1, 5):
    plt.plot(df['timestamp'], df[f'scenario_{i}'], marker='o', label=f'Scenario {i}')

plt.xlabel('Experiment Run')
plt.ylabel('Error Rate')
plt.title('Model Performance Across Multiple Experiments')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/performance_comparison.png', dpi=150)
plt.show()
```

### Generate Comparative Report

```python
import os
from datetime import datetime

def generate_comparative_report(results_dir='results'):
    experiments = sorted([d for d in os.listdir(results_dir) 
                         if os.path.isdir(os.path.join(results_dir, d))])

    report = []
    report.append("=" * 80)
    report.append("COMPARATIVE EXPERIMENT REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Experiments: {len(experiments)}")
    report.append("=" * 80)
    report.append("")

    for exp in experiments:
        summary_file = f'{results_dir}/{exp}/experiment_summary_{exp}.txt'
        if os.path.exists(summary_file):
            report.append(f"Experiment: {exp}")
            report.append("-" * 80)
            with open(summary_file, 'r') as f:
                lines = f.readlines()
                in_summary = False
                for line in lines:
                    if 'Summary of Results:' in line:
                        in_summary = True
                        continue
                    if in_summary and line.strip().startswith('Scenario'):
                        report.append("  " + line.strip())
                    if '=' in line and in_summary:
                        break
            report.append("")

    report.append("=" * 80)

    report_path = f'{results_dir}/comparative_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Comparative report saved to: {report_path}")

# Usage
generate_comparative_report()
```

---

## 📝 Best Practices

### Organizing Experiments

1. **Run experiments systematically**: Change one hyperparameter at a time
2. **Keep notes**: Document what you changed in a separate log file
3. **Name meaningfully**: Consider adding custom prefixes to result directories
4. **Archive old results**: Move completed experiment groups to subdirectories

### Storage Management

```bash
# Check total size of results
du -sh results/

# Find old experiments (older than 30 days)
find results/ -type d -mtime +30

# Archive old experiments
tar -czf results_archive_$(date +%Y%m%d).tar.gz results/*/

# Clean up archived experiments
# (Only after verifying archive is complete!)
```

### Documentation

Create a `results/experiment_log.md` to track experiments:

```markdown
# Experiment Log

## 2025-11-26

### Run 1 (14:30:25)
- **Purpose**: Baseline run with default hyperparameters
- **Changes**: None
- **Results**: See 20251126_143025/
- **Notes**: Good baseline performance

### Run 2 (15:01:32)
- **Purpose**: Test impact of increased epochs
- **Changes**: num_epochs = 20 (was 10)
- **Results**: See 20251126_150132/
- **Notes**: Slight improvement in Scenario 3
```

---

## 🚨 Troubleshooting

### Missing Files

**Issue:** Expected confusion matrix or summary file missing

**Solutions:**
1. Check if training completed successfully (no errors in terminal)
2. Verify results directory permissions
3. Ensure disk space is available
4. Re-run experiment if necessary

### Cannot Open Images

**Issue:** Confusion matrix PNG files won't open

**Solutions:**
1. Verify file size is not 0 bytes: `ls -lh results/*/confusion_matrix*.png`
2. Check file integrity: Try opening with different image viewer
3. Regenerate by re-running experiment

### Compare Results Not Matching

**Issue:** Results differ between runs with same parameters

**Causes:**
- Random initialization varies between runs
- Random train/test split varies
- GPU non-determinism (if using CUDA)

**Solutions:**
- Set random seeds for reproducibility (add to `train.py`)
- Average results over multiple runs
- Document variance in experiment logs

---

## 📚 Related Documentation

- **Main README**: [../README.md](../README.md) - Project overview and setup
- **Data README**: [../data/README.md](../data/README.md) - Dataset information
- **Source Code**: [../src/](../src/) - Training and preprocessing scripts

---

## 💡 Tips

- **Regular backups**: Keep important experiment results backed up
- **Version control**: Git commit code changes before running new experiments
- **Comparison baseline**: Always keep your first successful run as a baseline
- **Document hyperparameters**: Note any configuration changes in experiment logs
- **Visualize trends**: Create plots comparing multiple experiment runs
- **Share results**: Use confusion matrices in presentations and reports

---

**Need Help?** Check the main [README](../README.md) or open an [issue](https://github.com/julia-shtal/cnn-fashion-mnist/issues).
