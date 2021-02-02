(ns try-transform.main
  (:require [tablecloth.api :as tc]
            [tech.v3.ml :as ml]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset :as ds]
            [tech.v3.protocols.dataset :as proto-ds]
            [tech.v3.libs.xgboost]))






;;;  adapt exiting function to be transform compliant
(defn drop-rows [pipeline-context rows-selector]
  (assoc pipeline-context
         :dataset
         (tc/drop-rows  (:dataset pipeline-context) rows-selector))
  )

(defn categorical->number [pipeline-context filter-fn-or-dataset]
  (assoc pipeline-context
         :dataset
         (ds/categorical->number (:dataset pipeline-context) filter-fn-or-dataset))
  )

(defn set-inference-target [pipeline-context target-name-or-seq]
  (assoc pipeline-context
         :dataset
         (ds-mod/set-inference-target (:dataset pipeline-context) target-name-or-seq)))

(defn train-or-predict [pipeline-context id options ]
  (let [ds (:dataset pipeline-context)
        result (case (:mode pipeline-context)
                 :fit (ml/train ds options)
                 :transform (ml/predict ds (get pipeline-context id) )
                 )]
    (assoc  pipeline-context id result)))


(comment


  (def ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv"))



;;;  just fo demo, train=test
  (def train-ds ds)
  (def test-ds ds)


  (defn run-pipeline [ds context]
    (-> (merge context {:dataset ds})
        (drop-rows 2) ;; example for arbitray tabclecoth transformation
        (categorical->number cf/categorical)
        (set-inference-target "species")
        (train-or-predict :xgboost-hinge {:model-type :xgboost/binary-hinge-loss})
        (train-or-predict :xgboost-class {:model-type :xgboost/classification})
        ))


  ;; fit thge pipeline (including train)
  (def fit-result
    (run-pipeline train-ds {:mode :fit}))

;;;  transform (= predict ) on test
  (def transform-result
    (run-pipeline test-ds (merge fit-result {:mode :transform})))


  )
