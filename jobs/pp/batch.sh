tf=1.0
trf=1.0

./jobs/pp/primer.sh 'adult' 'sgb' $tf $trf 3 300 'short'
./jobs/pp/primer.sh 'adult' 'svm' $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'bank_marketing' 'sgb' $tf $trf 3 300 'short'
./jobs/pp/primer.sh 'bank_marketing' 'svm' $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'bean' 'xgb' $tf $trf 3 300 'short'
./jobs/pp/primer.sh 'bean' 'sgb' $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'compas' 'sgb' $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'credit_card' 'xgb' $tf $trf 3 1440 'short'
./jobs/pp/primer.sh 'credit_card' 'sgb' $tf $trf 3 1440 'short'
./jobs/pp/primer.sh 'credit_card' 'svm' $tf $trf 3 1440 'short'

./jobs/pp/primer.sh 'diabetes' 'xgb' $tf $trf 7 600  'short'
./jobs/pp/primer.sh 'diabetes' 'sgb' $tf $trf 7 600  'short'
./jobs/pp/primer.sh 'diabetes' 'knn' $tf $trf 7 600  'short'
./jobs/pp/primer.sh 'diabetes' 'svm' 0.1 $trf 7 1440 'short'
./jobs/pp/primer.sh 'diabetes' 'mlp' $tf $trf 7 600  'short'

./jobs/pp/primer.sh 'flight_delays' 'cb'   $tf $trf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'xgb'  $tf $trf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'sgb'  $tf $trf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'skrf' $tf $trf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'lr'   $tf $trf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'dt'   $tf $trf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'knn'  $tf $trf 7 600  'short'
./jobs/pp/primer.sh 'flight_delays' 'svm'  0.1 $trf 7 1440 'short'
./jobs/pp/primer.sh 'flight_delays' 'mlp'  $tf $trf 7 600  'short'

./jobs/pp/primer.sh 'msd' 'lgb'  $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'msd' 'cb'   $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'msd' 'xgb'  $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'msd' 'skrf' 0.25 $trf 7 1440 'short'
./jobs/pp/primer.sh 'msd' 'lr'   $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'msd' 'dt'   $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'msd' 'knn'  0.1  $trf 7 1440 'short'
./jobs/pp/primer.sh 'msd' 'svm'  0.1  $trf 7 1440 'short'
./jobs/pp/primer.sh 'msd' 'mlp'  $tf  $trf 7 1440 'short'

./jobs/pp/primer.sh 'htru2' 'sgb' $tf $trf 3 1440 'short'

./jobs/pp/primer.sh 'life' 'sgb' $tf $trf 3 1440 'short'

./jobs/pp/primer.sh 'naval' 'sgb' $tf $trf 3 1440 'short'

./jobs/pp/primer.sh 'no_show' 'xgb' $tf $trf 7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'sgb' $tf $trf 7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'lr'  $tf $trf 7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'dt'  $tf $trf 7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'knn' 0.1 $trf 7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'svm' 0.1 $trf 7 1440 'short'
./jobs/pp/primer.sh 'no_show' 'mlp' $tf $trf 7 1440 'short'

./jobs/pp/primer.sh 'power' 'sgb' $tf $trf 3 1440 'short'

./jobs/pp/primer.sh 'obesity' 'xgb' $tf $trf 7 1440 'short'
./jobs/pp/primer.sh 'obesity' 'sgb' $tf $trf 7 1440 'short'

./jobs/pp/primer.sh 'protein' 'xgb' $tf $trf 3 300 'short'
./jobs/pp/primer.sh 'protein' 'sgb' $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'spambase' 'sgb' $tf $trf 3 300 'short'
./jobs/pp/primer.sh 'spambase' 'skrf' $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'surgical' 'sgb' $tf $trf 3 300 'short'
./jobs/pp/primer.sh 'surgical' 'svm' $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'synth_binary' 'svm' $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'synth_multiclass' 'svm' $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'synth_regression' 'svm' $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'twitter' 'lgb'  $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'sgb'  $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'cb'   $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'xgb'  $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'skrf' $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'lr'   $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'dt'   $tf  $trf 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'knn'  0.1  $trf 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'svm'  0.01 0.25 7 1440 'short'
./jobs/pp/primer.sh 'twitter' 'mlp'  $tf  $trf 7 1440 'short'

./jobs/pp/primer.sh 'vaccine' 'sgb'  $tf $trf 3 300 'short'
./jobs/pp/primer.sh 'vaccine' 'skrf' $tf $trf 3 300 'short'
./jobs/pp/primer.sh 'vaccine' 'svm'  $tf $trf 3 300 'short'

./jobs/pp/primer.sh 'wine' 'sgb'  $tf $trf 3 300 'short'
