import tensorflow as tf

# ================= CONFIG =================
TRAIN_DIR = "gating/train"
VAL_DIR   = "gating/val"

IMG_SIZE = (192, 192)
BATCH = 32
EPOCHS = 20
LR = 1e-3
# =========================================

# --------- DATASETS ---------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="binary"
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# --------- AUGMENTATIONS ---------
augment = tf.keras.Sequential([
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.25),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
], name="augmentation")

# --------- BACKBONE ---------
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False   # congelado → estable y rápido

# --------- MODELO ---------
inp = tf.keras.Input((*IMG_SIZE, 3))
x = augment(inp)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inp, out, name="gating_model")

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# --------- CALLBACKS ---------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )
]

# --------- TRAIN ---------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --------- SAVE ---------
model.save("gating_model_final.keras")
print("✅ Guardado: gating_model_final.keras")