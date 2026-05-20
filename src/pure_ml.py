import os
import numpy as np
import pandas as pd

class PureStandardScaler:
    """Pure NumPy implementation of StandardScaler to avoid leaks."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Prevent division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PureLogisticRegression:
    """Pure NumPy Logistic Regression with L1/L2 regularization."""
    def __init__(self, penalty='l2', C=1.0, lr=0.05, epochs=3000, random_state=42):
        self.penalty = penalty
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.w = None
        self.b = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        m, n = X.shape
        
        # Consistent weights initialization
        rng = np.random.default_rng(self.random_state)
        self.w = rng.normal(0, 0.01, n)
        self.b = 0.0
        
        lambda_val = 1.0 / (self.C + 1e-8)
        
        for _ in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            p = self._sigmoid(z)
            
            dw = np.dot(X.T, p - y) / m
            db = np.sum(p - y) / m
            
            # Apply L1 or L2 regularization penalty
            if self.penalty == 'l2':
                dw += lambda_val * self.w / m
            elif self.penalty == 'l1':
                dw += lambda_val * np.sign(self.w) / m
                
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        z = np.dot(X, self.w) + self.b
        p = self._sigmoid(z)
        return np.column_stack((1.0 - p, p))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class Node:
    """Recursive decision node inside DecisionTree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class PureDecisionTreeClassifier:
    """Pure NumPy Decision Tree Classifier using Gini impurity."""
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.rng = np.random.default_rng(random_state)

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0.0
        counts = np.bincount(y)
        p = counts / m
        return 1.0 - np.sum(p**2)

    def _split(self, X, y, feature, threshold):
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        return left_idx, right_idx

    def _best_split(self, X, y, feature_indices):
        best_gain = -1.0
        best_feat = None
        best_thresh = None
        
        current_gini = self._gini(y)
        m = len(y)
        
        for feat in feature_indices:
            feat_values = X[:, feat]
            thresholds = np.unique(feat_values)
            
            # Sub-sample thresholds if features are continuous to speed up fit
            if len(thresholds) > 15:
                thresholds = np.percentile(feat_values, np.arange(10, 100, 10))
                
            for thresh in thresholds:
                left_idx, right_idx = self._split(X, y, feat, thresh)
                m_l, m_r = np.sum(left_idx), np.sum(right_idx)
                
                if m_l == 0 or m_r == 0:
                    continue
                    
                gini_l = self._gini(y[left_idx])
                gini_r = self._gini(y[right_idx])
                
                child_gini = (m_l / m) * gini_l + (m_r / m) * gini_r
                gain = current_gini - child_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
                    
        return best_feat, best_thresh

    def _build_tree(self, X, y, depth=0):
        m, n = X.shape
        num_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or 
            num_labels == 1 or 
            m < self.min_samples_split):
            leaf_val = np.bincount(y).argmax() if len(y) > 0 else 0
            return Node(value=leaf_val)
            
        if self.max_features is None:
            feature_indices = np.arange(n)
        elif self.max_features == 'sqrt':
            n_features = max(1, int(np.sqrt(n)))
            feature_indices = self.rng.choice(n, n_features, replace=False)
        else:
            feature_indices = np.arange(n)
            
        best_feat, best_thresh = self._best_split(X, y, feature_indices)
        
        if best_feat is None:
            leaf_val = np.bincount(y).argmax() if len(y) > 0 else 0
            return Node(value=leaf_val)
            
        left_idx, right_idx = self._split(X, y, best_feat, best_thresh)
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build_tree(X, y)
        return self

    def _predict_row(self, node, row):
        if node.is_leaf():
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(node.left, row)
        return self._predict_row(node.right, row)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_row(self.root, row) for row in X])

    def _predict_proba_row(self, node, row, num_classes=2):
        if node.is_leaf():
            probs = np.zeros(num_classes)
            probs[node.value] = 1.0
            return probs
        if row[node.feature] <= node.threshold:
            return self._predict_proba_row(node.left, row, num_classes)
        return self._predict_proba_row(node.right, row, num_classes)

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.array([self._predict_proba_row(self.root, row) for row in X])


class PureRandomForestClassifier:
    """Pure NumPy Random Forest Classifier with bootstrap and max_features support."""
    def __init__(self, n_estimators=50, max_depth=5, min_samples_split=2, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.rng = np.random.default_rng(random_state)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        m = len(y)
        self.trees = []
        
        for i in range(self.n_estimators):
            bootstrap_indices = self.rng.choice(m, m, replace=True)
            X_b = X[bootstrap_indices]
            y_b = y[bootstrap_indices]
            
            tree = PureDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state + i
            )
            tree.fit(X_b, y_b)
            self.trees.append(tree)
            
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_probs, axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    @property
    def feature_importances_(self):
        """Clinical Gini importance equivalent calculated dynamically."""
        n_features = 13
        importances = np.zeros(n_features)
        
        def traverse(node, depth=1):
            if node.is_leaf():
                return
            # Give higher priority weight to features selected higher up the decision tree
            importances[node.feature] += 1.0 / (depth**2)
            traverse(node.left, depth + 1)
            traverse(node.right, depth + 1)
            
        for tree in self.trees:
            traverse(tree.root)
            
        sum_imp = np.sum(importances)
        if sum_imp > 0:
            importances /= sum_imp
        else:
            importances = np.ones(n_features) / n_features
            
        return importances


class PureDecisionTreeRegressor:
    """Pure NumPy Decision Tree Regressor using variance reduction."""
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.rng = np.random.default_rng(random_state)

    def _variance(self, y):
        if len(y) == 0:
            return 0.0
        return np.var(y)

    def _split(self, X, y, feature, threshold):
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        return left_idx, right_idx

    def _best_split(self, X, y, feature_indices):
        best_gain = -1.0
        best_feat = None
        best_thresh = None
        
        current_var = self._variance(y)
        m = len(y)
        
        for feat in feature_indices:
            feat_values = X[:, feat]
            thresholds = np.unique(feat_values)
            
            # Sub-sample thresholds if features are continuous to speed up fit
            if len(thresholds) > 15:
                thresholds = np.percentile(feat_values, np.arange(10, 100, 10))
                
            for thresh in thresholds:
                left_idx, right_idx = self._split(X, y, feat, thresh)
                m_l, m_r = np.sum(left_idx), np.sum(right_idx)
                
                if m_l == 0 or m_r == 0:
                    continue
                    
                var_l = self._variance(y[left_idx])
                var_r = self._variance(y[right_idx])
                
                child_var = (m_l / m) * var_l + (m_r / m) * var_r
                gain = current_var - child_var
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
                    
        return best_feat, best_thresh

    def _build_tree(self, X, y, depth=0):
        m, n = X.shape
        
        if (depth >= self.max_depth or 
            m < self.min_samples_split or
            np.all(y == y[0])):
            leaf_val = float(np.mean(y)) if len(y) > 0 else 0.0
            return Node(value=leaf_val)
            
        if self.max_features is None:
            feature_indices = np.arange(n)
        elif self.max_features == 'sqrt':
            n_features = max(1, int(np.sqrt(n)))
            feature_indices = self.rng.choice(n, n_features, replace=False)
        else:
            feature_indices = np.arange(n)
            
        best_feat, best_thresh = self._best_split(X, y, feature_indices)
        
        if best_feat is None:
            leaf_val = float(np.mean(y)) if len(y) > 0 else 0.0
            return Node(value=leaf_val)
            
        left_idx, right_idx = self._split(X, y, best_feat, best_thresh)
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        self.root = self._build_tree(X, y)
        return self

    def _predict_row(self, node, row):
        if node.is_leaf():
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(node.left, row)
        return self._predict_row(node.right, row)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_row(self.root, row) for row in X])


class PureGradientBoostingClassifier:
    """Pure NumPy Gradient Boosting Classifier with decision trees as base learners."""
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_split=2, max_features=None, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.classes_ = np.array([0, 1])
        self.F_0 = 0.0

    def _update_leaf_values(self, node, X, r, p, indices):
        if node.is_leaf():
            # Calculate Newton-Raphson update for log-loss
            numerator = np.sum(r[indices])
            denominator = np.sum(p[indices] * (1.0 - p[indices]))
            if denominator < 1e-10:
                node.value = 0.0
            else:
                node.value = float(numerator / denominator)
            return
        
        # Split indices
        left_mask = X[indices, node.feature] <= node.threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        
        if len(left_indices) > 0:
            self._update_leaf_values(node.left, X, r, p, left_indices)
        if len(right_indices) > 0:
            self._update_leaf_values(node.right, X, r, p, right_indices)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        m, n = X.shape
        
        # Initialize F0(x) with log-odds
        p_init = np.mean(y)
        p_init = np.clip(p_init, 1e-15, 1.0 - 1e-15)
        self.F_0 = float(np.log(p_init / (1.0 - p_init)))
        
        # Current raw scores
        F = np.full(m, self.F_0)
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Compute probabilities
            p = 1.0 / (1.0 + np.exp(-F))
            
            # Compute pseudo-residuals (gradient of log-loss)
            r = y - p
            
            # Fit regressor tree to pseudo-residuals
            tree = PureDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state + i
            )
            tree.fit(X, r)
            
            # Update leaf values using Newton-Raphson step
            all_indices = np.arange(m)
            self._update_leaf_values(tree.root, X, r, p, all_indices)
            
            # Update scores F
            F += self.learning_rate * tree.predict(X)
            
            self.trees.append(tree)
            
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        m = X.shape[0]
        F = np.full(m, self.F_0)
        
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
            
        p = 1.0 / (1.0 + np.exp(-F))
        return np.column_stack((1.0 - p, p))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        """Variance reduction weighted importance equivalent calculated dynamically."""
        n_features = 13
        importances = np.zeros(n_features)
        
        def traverse(node, depth=1):
            if node.is_leaf():
                return
            if node.feature < n_features:
                importances[node.feature] += 1.0 / (depth**2)
            traverse(node.left, depth + 1)
            traverse(node.right, depth + 1)
            
        for tree in self.trees:
            traverse(tree.root)
            
        sum_imp = np.sum(importances)
        if sum_imp > 0:
            importances /= sum_imp
        else:
            importances = np.ones(n_features) / n_features
            
        return importances


def pure_stratified_kfold(y, n_splits=5, random_state=42):
    """Pure NumPy stratified split logic."""
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    
    class_indices = {}
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        class_indices[label] = idx
        
    folds = [[] for _ in range(n_splits)]
    
    for label, idxs in class_indices.items():
        for i, idx in enumerate(idxs):
            folds[i % n_splits].append(idx)
            
    splits = []
    for test_fold_idx in range(n_splits):
        test_indices = np.array(folds[test_fold_idx])
        train_indices = np.array([
            idx for fold_idx, fold in enumerate(folds) 
            if fold_idx != test_fold_idx for idx in fold
        ])
        splits.append((train_indices, test_indices))
        
    return splits


def pure_roc_auc_score(y_true, y_score):
    """Calculates Area Under the ROC curve using standard trapezoidal rule integration."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
        
    tp = 0
    fp = 0
    auc = 0.0
    
    for val in y_true_sorted:
        if val == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
            
    return auc / (n_pos * n_neg)


def pure_roc_curve(y_true, y_score):
    """Calculates false positive rates and true positive rates for ROC curves."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    tpr = np.insert(tpr, 0, 0.0)
    fpr = np.insert(fpr, 0, 0.0)
    
    return fpr, tpr, np.insert(y_score_sorted, 0, 1.0)


def pure_confusion_matrix(y_true, y_pred):
    """Generates 2x2 confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[tn, fp], [fn, tp]])


def pure_stratified_train_test_split(X, y, test_size=0.20, random_state=42):
    """Pure NumPy stratified split logic for DataFrames or NumPy arrays to prevent leaks."""
    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()
    
    rng = np.random.default_rng(random_state)
    
    train_indices = []
    test_indices = []
    
    for label in np.unique(y):
        idx = y[y == label].index.values.copy()
        rng.shuffle(idx)
        
        n_test = max(1, int(len(idx) * test_size))
        test_indices.extend(idx[:n_test])
        train_indices.extend(idx[n_test:])
        
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    
    # Shuffle splits to preserve randomness
    train_shuffle = rng.permutation(len(train_indices))
    X_train = X_train.iloc[train_shuffle]
    y_train = y_train.iloc[train_shuffle]
    
    test_shuffle = rng.permutation(len(test_indices))
    X_test = X_test.iloc[test_shuffle]
    y_test = y_test.iloc[test_shuffle]
    
    return X_train, X_test, y_train, y_test


def pure_grid_search_cv(model_class, param_grid, X, y, cv_splits):
    """Pure K-Fold cross validation and grid search hyperparameter tuning."""
    best_score = -1.0
    best_params = None
    
    # Generate all parameter combinations
    import itertools
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for params in param_combinations:
        scores = []
        for train_idx, test_idx in cv_splits:
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_test_fold, y_test_fold = X[test_idx], y[test_idx]
            
            # Initialize model with params
            model = model_class(**params)
            model.fit(X_train_fold, y_train_fold)
            
            # Calculate ROC-AUC on test fold
            probs = model.predict_proba(X_test_fold)[:, 1]
            auc = pure_roc_auc_score(y_test_fold, probs)
            scores.append(auc)
            
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            
    return best_params, best_score


