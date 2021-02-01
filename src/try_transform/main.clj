(ns try-transform.main
  (:require [tablecloth.api :as tc]
            [tech.v3.ml :as ml]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset :as ds]
            [tech.v3.protocols.dataset :as proto-ds]
            [tech.v3.libs.xgboost]))




;;;  ease the duality of map-dataset

(defn ->ds [ds-or-map]
  (get ds-or-map :dataset ds-or-map )

  )

(defn merge-or-assoc-ds
  ([id ds-or-map new-map]
   (if (satisfies? proto-ds/PColumnarDataset ds-or-map)
     (assoc new-map
            :dataset ds-or-map)
     (assoc ds-or-map id new-map)
     ))
  ([ds-or-map]
   (merge-or-assoc-ds ds-or-map {})
   )

  )

(defn associate-ds [ds-or-map ds]
  (if (satisfies? proto-ds/PColumnarDataset ds-or-map)
    {:dataset ds}
    (merge ds-or-map
           {:dataset ds}
           )
    )
  )


;;;  adapt exiting function to be transform compliant
(defn drop-rows [ds-or-map rows-selector]
  (associate-ds ds-or-map
                (tc/drop-rows  (->ds ds-or-map) rows-selector))
  )

(defn categorical->number [ds-or-map filter-fn-or-dataset]
  (associate-ds ds-or-map
   (ds/categorical->number (->ds ds-or-map) filter-fn-or-dataset))
  )

(defn set-inference-target [ds-or-map target-name-or-seq]
  (associate-ds ds-or-map
                (ds-mod/set-inference-target (->ds ds-or-map) target-name-or-seq)))

(defn train-or-predict [ds-or-map id options ]
  ;;;  this function requires map
  (let [ds (:dataset ds-or-map)
        result (case (:mode ds-or-map)
                 :fit (ml/train ds options)
                 :transform (ml/predict ds (get ds-or-map id) )
                 )]
    (merge-or-assoc-ds id ds-or-map result)))





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
      (train-or-predict :xgboost-class {:model-type :xgboost/classification})))


;; fit thge pipeline (including train)
(def fit-result
  (run-pipeline train-ds {:mode :fit}))

;;;  transform (= predict ) on test
(def transform-result
  (run-pipeline test-ds (merge fit-result {:mode :transform})))
