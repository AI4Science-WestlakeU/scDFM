export PYTHONPATH=./

python  src/script/run.py  \
--batch_size=48 \
--devices='0' \
--model_type=origin \
--lr=5e-5 \
--steps=200000 \
--data_name=norman \
--d_model=128 \
--eta_min=1e-6 \
--fusion_method=differential_perceiver \
--infer_top_gene=1000 \
--n_top_genes=5000 \
--result_path=./result/additive \
--perturbation_function=crisper \
--noise_type=Gaussian \
--mode=predict_y \
--gamma=0.5 \
--split_method=additive \
--use_mmd_loss \
--fold=1 \
--topk=30 \
--use_negative_edge \
