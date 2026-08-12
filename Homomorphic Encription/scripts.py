"""
ORQUESTACIÓN DEL ENTRENAMIENTO
==============================
Submuestreo aleatorio repetido (Monte Carlo): entrena un modelo por semilla
remezclando train/val, con el test fijo. El muestreo produce dos cosas:

  - el modelo/semilla con mejor accuracy de validación, que fija la
    inicialización de pesos del modelo definitivo;
  - la MEDIANA de las épocas en que la pérdida de validación fue óptima, que es
    el hiperparámetro que se transfiere al reentrenamiento final.

Después se reentrena con train+val durante ese número fijo de épocas. Guarda
modelo, configuración, pesos iniciales, scaler y las gráficas de comparación de
semillas e historial de entrenamiento.
"""

import copy
import json
import os
import statistics
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from Trainer import Trainer


def train_with_multiple_seeds(model, processor, seeds, test_size, val_size, pca_features,
                              learning_rate=0.001, epochs=100, output_dir='output_dir', rg=0):
    """
    Entrena el modelo con múltiples semillas y selecciona el mejor.

    Args:
        model: instancia de ConfigurableNN a entrenar (se reinicializa por semilla).
        processor: DataProcessor con los datos ya cargados.
        seeds: lista de semillas a probar.
        test_size, val_size: proporciones de test y validación sobre el total.
        pca_features: nº de componentes PCA (0 = sin PCA).
        learning_rate, epochs, rg: hiperparámetros de entrenamiento.
        output_dir: carpeta donde guardar modelo, config, pesos y scaler.

    Returns:
        (best_model, best_seed, best_test_dataset, results, epocas_finales)
        donde `epocas_finales` es la mediana de las épocas óptimas entre semillas,
        que es la que se ha usado para el reentrenamiento con train+val.
    """
    # Almacenar resultados
    results = []
    best_val_acc = 0
    best_model = None
    best_seed = None
    best_test_dataset = None
    best_initial_weights = None

    num_model = 0

    print(f"Se entrenaran {len(seeds)} modelos")


    for s in seeds:
        seed = s
        train_set, val_set, test_set = processor.split_data(test_size=test_size, val_size=val_size, seed=s)

        print(f"Semilla actual {seed}")
        #Reset de pesos: sembramos torch con la semilla para que la inicialización
        #sea reproducible y el reentrenamiento final parta de los mismos pesos
        torch.manual_seed(seed)
        model._initialize_weights()
        # Guardar pesos iniciales (deepcopy: state_dict devuelve referencias vivas)
        initial_weights = copy.deepcopy(model.state_dict())

        print(f'Procesando datos\n Conjunto validacion {val_size}\n Conjunto test {test_size}')
        train_loader, val_loader, test_loader = processor.transform_and_load(train_set=train_set,
                                                                            val_set=val_set, test_set=test_set, pca_features=pca_features,
                                                                            batch_size=32)
        
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                      learning_rate=learning_rate, rg=rg)
        # Entrenar
        
        print(f"   Entrenando modelo {num_model}")
        best_val_loss, final_val_acc, best_epoch = trainer.train(epochs=epochs,
                                                        early_stopping_patience=15,
                                                        verbose=False)

        print(f"   ✅ Completado modelo {num_model} - Val Accuracy: {final_val_acc:.2f}% "
              f"(época óptima: {best_epoch})")

        # Guardar resultados
        result = {
                'seed': seed,
                'val_accuracy': final_val_acc,
                'val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'train_history': {
                    'train_losses': trainer.train_losses,
                    'train_accuracies': trainer.train_accuracies,
                    'val_losses': trainer.val_losses,
                    'val_accuracies': trainer.val_accuracies
                }
            }
        #print(f"Resultados {result}")

        results.append(result)
        
        # Actualizar mejor modelo
        if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_model = model
                best_seed = seed
                best_test_dataset = test_loader
                best_initial_weights = initial_weights
                print(f"   🌟 ¡NUEVO MEJOR MODELO!")
    
        
        num_model +=1

    # Resumen final
    print(f"\n{'='*70}")
    print("📊 RESUMEN DE RESULTADOS")
    print(f"{'='*70}")
    
    for i, res in enumerate(results):
        marker = "🌟" if res['seed'] == best_seed else "  "
        print(f"{marker} Semilla {res['seed']}: Val Acc = {res['val_accuracy']:.2f}% "
              f"(época óptima {res['best_epoch']})")

    # Estadísticos del muestreo Monte Carlo. La media y la desviación describen
    # la estabilidad del entrenamiento; el máximo es solo el criterio de selección
    # y está sesgado al alza, así que no debe reportarse como rendimiento.
    accs = [r['val_accuracy'] for r in results]
    val_media = statistics.mean(accs)
    val_desviacion = statistics.pstdev(accs) if len(accs) > 1 else 0.0

    # Época del reentrenamiento: mediana entre semillas. Es más robusta que la
    # época de la mejor semilla, que depende de una sola partición.
    epocas = [r['best_epoch'] for r in results if r['best_epoch']]
    epocas_finales = max(1, int(statistics.median(epocas))) if epocas else epochs

    print(f"\n✨ MEJOR MODELO:")
    print(f"   Semilla: {best_seed}")
    print(f"   Val Accuracy: {best_val_acc:.2f}%")
    print(f"\n📐 MUESTREO MONTE CARLO ({len(results)} semillas):")
    print(f"   Val Accuracy media: {val_media:.2f}% ± {val_desviacion:.2f}")
    print(f"   Épocas óptimas: {epocas} → mediana {epocas_finales}")

    print(f" REENTRENAMOS MEJOR MODELO CON TRAIN Y VAL SET")

    train_set, val_set, test_set = processor.split_data(test_size=test_size, val_size=val_size, seed=best_seed)
    #Reset de pesos con la misma semilla: los pesos iniciales del reentrenamiento
    #son exactamente los mismos que los del mejor modelo de la búsqueda
    torch.manual_seed(best_seed)
    model._initialize_weights()

    train_loader, val_loader, test_loader = processor.transform_and_load(train_set=train_set,
                                                                            val_set=val_set, test_set=test_set, pca_features=pca_features,
                                                                            batch_size=32)
        
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                      learning_rate=learning_rate, rg=rg)

    rango_preactivaciones = trainer.last_train(epochs=epocas_finales)
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar información del experimento
    experiment_info = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'architecture': model.architecture,
        'input_size': model.input_size,
        'num_classes': model.num_classes,
        'num_seeds_tried': len(seeds),
        'best_seed': best_seed,
        'best_val_accuracy': best_val_acc,
        'val_accuracy_media': val_media,
        'val_accuracy_desviacion': val_desviacion,
        'epocas_optimas': epocas,
        'epocas_reentrenamiento': epocas_finales,
        'all_results': [{'seed': r['seed'], 'val_accuracy': r['val_accuracy'],
                         'best_epoch': r['best_epoch']}
                       for r in results],
        'features' : pca_features,
    }
    
    with open(os.path.join(output_dir, f'Nfeatures_{experiment_info["features"]}_experiment_info.json'), 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    # Guardar configuración del mejor modelo
    model_config = best_model.get_config()
    model_config['best_seed'] = best_seed
    model_config['best_val_accuracy'] = best_val_acc
    model_config['epocas_reentrenamiento'] = epocas_finales
    # Intervalo en el que PrediccionHomomorfica ajusta el polinomio que sustituye
    # a la ReLU: se mide sobre train+val con los pesos definitivos y se publica
    # con el modelo (None si la red no tiene capas ocultas)
    model_config['rango_preactivaciones'] = (list(rango_preactivaciones)
                                             if rango_preactivaciones else None)
    
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Guardar pesos iniciales del mejor modelo
    torch.save(best_initial_weights, os.path.join(output_dir, 'initial_weights.pth'))
    
    # Guardar scaler
    import pickle
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(processor.Scaler, f)

    print(f"\n💾 Archivos guardados en: {output_dir}/")
    print(f"   • model_config.json - Configuración del modelo")
    print(f"   • experiment_info.json - Info del experimento")
    print(f"   • initial_weights.pth - Pesos iniciales del mejor modelo")
    print(f"   • scaler.pkl - Normalizador de datos")

    return best_model, best_seed, best_test_dataset, results, epocas_finales


def save_final_model_and_plots(model, seed, results, test_dataset, output_dir='experiment_results'):
    """
    Guarda el modelo final, gráficas y prepara datasets para predicción
    """
    print(f"\n{'='*70}")
    print("💾 GUARDANDO MODELO FINAL Y RESULTADOS")
    print(f"{'='*70}")
    
    # Crear subdirectorios
    models_dir = os.path.join(output_dir, 'models')
    datasets_dir = os.path.join(output_dir, 'datasets')
    results_dir = os.path.join(output_dir, 'results')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # best_model.pth es el que cargan PrediccionPlana y PrediccionHomomorfica
    torch.save(model.state_dict(), os.path.join(models_dir, 'best_model.pth'))
    print(f"✅ Modelo guardado: {models_dir}/best_model.pth")
    
    # Guardar test dataset (para predicción)
    # Esto funciona aunque sea un DataLoader
    all_X = []
    all_y = []
    for batch_X, batch_y in test_dataset:
        all_X.append(batch_X)
        all_y.append(batch_y)

    test_X = torch.cat(all_X, dim=0)
    test_y = torch.cat(all_y, dim=0)

    torch.save({
        'X': test_X,
        'y': test_y
    }, os.path.join(datasets_dir, 'test_data.pt'))
    print(f"✅ Test dataset guardado: {datasets_dir}/test_data.pt")
    

    # Crear gráficas de comparación de todas las semillas
    plot_all_seeds_comparison(results, results_dir)
    
    # Crear gráfica del mejor modelo
    best_result = [r for r in results if r['seed'] == seed][0]
    plot_best_model_history(best_result, results_dir)
    
    print(f"\n✅ Todo guardado correctamente en: {output_dir}/")


def plot_all_seeds_comparison(results, save_dir):
    """Crea gráfica comparando todas las semillas"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Comparación de accuracy final
    seeds = [r['seed'] for r in results]
    val_accs = [r['val_accuracy'] for r in results]
    
    colors = ['gold' if acc == max(val_accs) else 'skyblue' for acc in val_accs]
    
    ax1.bar(range(len(seeds)), val_accs, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Semilla', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Val Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de Accuracy por Semilla', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(seeds)))
    ax1.set_xticklabels(seeds)
    ax1.grid(axis='y', alpha=0.3)
    
    # Añadir valores
    for i, acc in enumerate(val_accs):
        ax1.text(i, acc + 0.5, f'{acc:.2f}%', ha='center', fontweight='bold')
    
    # Evolución de training de cada semilla
    for i, result in enumerate(results):
        history = result['train_history']
        label = f"Seed {result['seed']}" + (" (mejor)" if result['val_accuracy'] == max(val_accs) else "")
        linewidth = 3 if result['val_accuracy'] == max(val_accs) else 1.5
        alpha = 1.0 if result['val_accuracy'] == max(val_accs) else 0.6
        
        ax2.plot(history['val_accuracies'], label=label, linewidth=linewidth, alpha=alpha)
    
    ax2.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Val Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Evolución del Entrenamiento', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'seeds_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Comparación de semillas: {save_path}")


def plot_best_model_history(best_result, save_dir):
    """Crea gráfica detallada del mejor modelo"""
    history = best_result['train_history']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_losses'], label='Entrenamiento', linewidth=2.5, color='#3498db')
    ax1.plot(history['val_losses'], label='Validación', linewidth=2.5, color='#e74c3c')
    ax1.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pérdida (Loss)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Pérdida - Mejor Modelo (Semilla {best_result["seed"]})', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_accuracies'], label='Entrenamiento', linewidth=2.5, color='#3498db')
    ax2.plot(history['val_accuracies'], label='Validación', linewidth=2.5, color='#e74c3c')
    ax2.set_xlabel('Época', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precisión (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Precisión - Mejor Modelo (Semilla {best_result["seed"]})', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Historial del mejor modelo: {save_path}")


# Este módulo no se ejecuta directamente: usa main.py para un experimento
# individual o ejecutar_experimentos.py para la batería completa.
