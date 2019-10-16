python3 run_grid_cern_tuning.py \
      -c user.jodafons.cnn_short_test \
      -d user.jodafons.data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_V97_et0_eta0.npz \
      -r user.jodafons.data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et0_eta0.ref.pic.gz \
      --containerImage docker://jodafons/gpu-base:latest  \
      -o user.jodafons.data17_13TeV.Allperiods.sgn.probes_lhmedium_EG1.bkg.VProbes_EG7.mlp.ringer_v8_et0_eta0.cnn_short_test \
      -j /code/saphyra/Analysis/RingerNote_2018/tunings/v8/job_tuning.py \
      --site AUTO \
      --et 0 \
      --eta 0 \
      --user jodafons \
      --url "postgres://ringer:6sJ09066sV1990;6@postgres-ringer-db.cahhufxxnnnr.us-east-2.rds.amazonaws.com/ringer" \
      --dry-run \
      --njobs 10
