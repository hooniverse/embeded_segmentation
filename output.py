def loss_and_dice():
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model dice_coef')
    plt.ylabel('dice_coef')
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'], loc='lower right')

    plt.tight_layout()
    plt.show()
    
def visualize_results(model, X, y, num_samples=3):
    predictions = model.predict(X[:num_samples])
    predictions = (predictions > 0.5).astype(np.float32)
    
    plt.figure(figsize=(4 * num_samples, 12))
    for i in range(num_samples):
        # 원본 이미지 출력
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(X[i])
        plt.title('Original Image')
        plt.axis('off')
        
        # 실제 마스크 출력
        plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(y[i, :, :, 0], cmap='gray')
        plt.title(f'True Mask\nDice: {dice_coef(y[i:i+1], predictions[i:i+1]).numpy():.4f}')
        plt.axis('off')
        
        # 예측 마스크 출력
        plt.subplot(3, num_samples, i + 1 + 2 * num_samples)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def calculate_flops(model, input_shape=(1, 128, 128, 3)):
    # 모델을 ConcreteFunction으로 변환
    concrete_func = tf.function(lambda x: model(x))
    concrete_func = concrete_func.get_concrete_function(
        tf.TensorSpec(input_shape, tf.float32)
    )

    # Graph를 Frozen하여 FLOPs 계산
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_graph = frozen_func.graph

    # TensorFlow Profiler를 사용하여 FLOPs 계산
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # tf.compat.v1.profiler을 사용하여 계산
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_graph,
        run_meta=run_meta,
        options=opts
    )

    return flops.total_float_ops  # Total FLOPs