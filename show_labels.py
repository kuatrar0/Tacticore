#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show the 9 attacker and 31 victim labels from the improved model
"""

import pandas as pd

def show_model_labels():
    """Show all labels used in the improved model."""
    print("=" * 60)
    print("ETIQUETAS DEL MODELO MEJORADO")
    print("=" * 60)
    
    # Load CSV data
    df = pd.read_csv('features_labeled_context (2).csv')
    print(f"Total samples: {len(df)}")
    
    def parse_labels(label_string):
        if pd.isna(label_string) or label_string == '':
            return []
        return [label.strip() for label in str(label_string).split(',')]
    
    # Parse labels
    attacker_labels = df['attacker_label'].apply(parse_labels)
    victim_labels = df['victim_label'].apply(parse_labels)
    
    # Get unique labels (excluding redundant)
    redundant_attacker = {'is_alive', 'visible'}
    redundant_victim = {'is_alive'}
    
    attacker_unique = set()
    victim_unique = set()
    
    for labels in attacker_labels:
        filtered = [l for l in labels if l not in redundant_attacker]
        attacker_unique.update(filtered)
    
    for labels in victim_labels:
        filtered = [l for l in labels if l not in redundant_victim]
        victim_unique.update(filtered)
    
    print(f"\nATACANTE ETIQUETAS ({len(attacker_unique)}):")
    print("Estas son las fortalezas que puede tener el atacante:")
    for i, label in enumerate(sorted(attacker_unique), 1):
        print(f"  {i:2d}. {label}")
    
    print(f"\nVICTIMA ETIQUETAS ({len(victim_unique)}):")
    print("Estas son los errores que puede cometer la victima:")
    for i, label in enumerate(sorted(victim_unique), 1):
        print(f"  {i:2d}. {label}")
    
    print(f"\nFRECUENCIA DE FORTALEZAS DEL ATACANTE:")
    attacker_counts = {}
    for labels in attacker_labels:
        for label in labels:
            if label not in redundant_attacker:
                attacker_counts[label] = attacker_counts.get(label, 0) + 1
    
    for label, count in sorted(attacker_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count/len(df)*100
        print(f"  {label:20s}: {count:3d} kills ({percentage:5.1f}%)")
    
    print(f"\nFRECUENCIA DE ERRORES DE VICTIMA:")
    victim_counts = {}
    for labels in victim_labels:
        for label in labels:
            if label not in redundant_victim:
                victim_counts[label] = victim_counts.get(label, 0) + 1
    
    for label, count in sorted(victim_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count/len(df)*100
        print(f"  {label:20s}: {count:3d} kills ({percentage:5.1f}%)")
    
    print(f"\nRESUMEN:")
    print(f"  - {len(attacker_unique)} etiquetas de atacante (fortalezas)")
    print(f"  - {len(victim_unique)} etiquetas de victima (errores)")
    print(f"  - Total: {len(attacker_unique) + len(victim_unique)} modelos entrenados")
    print(f"  - Etiquetas redundantes excluidas: is_alive, visible")

if __name__ == "__main__":
    show_model_labels()
