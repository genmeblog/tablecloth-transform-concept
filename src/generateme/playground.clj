(ns generateme.playground
  (:require [tablecloth.api :as tc]
            [clojure.string :as str]
            [tech.v3.ml :as ml]
            [clojure.set :as set]
            [tech.v3.dataset :as ds]))

;; let's define some new ML objects to for training/prediction

(defrecord CatToId [forward inverse])

(defn- cat->id
  [data]
  (let [d (zipmap data (range))]
    (->CatToId d (set/map-invert d))))

(defn- train-cat->id
  [ds column-selector]
  (->> column-selector
       (tc/select-columns ds)
       (tc/columns)
       (reduce concat)
       (distinct)
       (cat->id)))

(defn- predict-cat->id
  [ds column-selector {:keys [prefix model]}]
  (let [forward (:forward model)]
    (reduce (fn [ds nm]
              (tc/add-or-replace-column ds nm (map forward (tc/column ds nm)))) ds (tc/column-names ds column-selector))))

(keys (:forward (train-cat->id fifa21 "nationality")))

(predict-cat->id fifa21 "nationality" {:model (train-cat->id fifa21 "nationality")})

(ml/define-model! :generateme/)

;; fifa21 dataset: https://www.kaggle.com/aayushmishra1512/fifa-2021-detailed-eda

(def fifa21 
  (-> (tc/dataset "resources/fifa21_kaggle.csv" {:separator ";"}) ;; load
      (tc/update-columns "position" (partial map #(str/split % #"\|"))) ;; convert position to a sequence
      (tc/unroll "position") ;; create artificial rows to have each position separated
      (tc/add-or-replace-column :check 1) ;; add artificial column which is our target for reshaping
      (tc/pivot->wider "position" :check {:drop-missing? false}) ;; use position as a column name and `:check` column as value
      (tc/drop-columns ["player_id", "name"]) ;; remove unnecessary columns
      (tc/replace-missing :all :value 0))) ;; replace missing after reshaping with `0`

(tc/info fifa21)


(tc/clone (map inc (range 10)))
