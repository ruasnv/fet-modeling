"""PyTorchCrossValidator(

    run_cv()

    )
USE-CASE:
 cv_runner = PyTorchCrossValidator(
            device=device,
            criterion=nn.MSELoss(),
            scaler_x=dp.get_scalers()[0],
            scaler_y=dp.get_scalers()[1],
            features_for_model=dp.get_features_for_model()
        )

        summary_df, detailed_df = cv_runner.run_cv(
            model_config=model_config_cv,
            X_cv_scaled=dp.get_cv_data()[0],
            y_cv_scaled=dp.get_cv_data()[1],
            X_cv_original_df=dp.get_cv_data()[2],
            cv_fold_indices=dp.get_cv_data()[3],
            num_epochs=settings.get("nn_number_epochs"),
            batch_size=settings.get("NN_BATCH_SIZE"),
            model_save_base_dir = os.path.join(TRAINED_MODEL_DIR, 'k_fold_models'),
            report_output_dir=REPORT_OUTPUT_DIR,
            skip_if_exists=SKIP_TRAINING_IF_EXISTS
        )
"""