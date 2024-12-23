def load(data_dir, img_size=(128,128)):
    images = []
    masks = []

    for img_file in os.listdir(data_dir):
        if '_mask' not in img_file:
            img_path = os.path.join(data_dir, img_file)
            mask_path = os.path.join(data_dir, img_file.replace('.jpg', '_mask.jpg'))

            img = load_img(img_path, target_size=img_size)
            mask = load_img(mask_path, target_size=img_size, color_mode='grayscale')

            img = img_to_array(img) / 255.0
            mask = img_to_array(mask) / 255.0

            images.append(img)
            masks.append(mask)

    return np.array(images), np.array(masks)


def augment(images, masks, num_aug=2):
    
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    aug_images = []
    aug_masks = []
    
    # 원본 데이터 추가
    aug_images.extend(images)
    aug_masks.extend(masks)
    
    # 각 이미지에 대해 증강 데이터 생성
    for idx in range(len(images)):
        image = images[idx]
        mask = masks[idx]
        
        # 이미지와 마스크를 4D 텐서로 변환 (BatchSize=1)
        img_tensor = np.expand_dims(image, 0)
        mask_tensor = np.expand_dims(mask, 0)
        
        # 랜덤 시드 설정 (이미지와 마스크가 동일하게 변형되도록)
        seed = np.random.randint(1000)
        
        # 지정된 수만큼 증강 데이터 생성
        for _ in range(num_aug):
            # 이미지 증강
            datagen.fit(img_tensor)
            aug_img_iter = datagen.flow(img_tensor, seed=seed, batch_size=1)
            aug_img = next(aug_img_iter)[0]
            
            # 마스크 증강 (동일한 변환 적용)
            datagen.fit(mask_tensor)
            aug_mask_iter = datagen.flow(mask_tensor, seed=seed, batch_size=1)
            aug_mask = next(aug_mask_iter)[0]
            
            aug_images.append(aug_img)
            aug_masks.append(aug_mask)
    
    return np.array(aug_images), np.array(aug_masks)