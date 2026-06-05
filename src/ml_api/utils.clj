(ns ml-api.utils
  "Utility function for the ML-API Application."
  (:require
   [flambo.sql :as sql])
  (:import
   [org.apache.spark.ml.linalg Vector]
   [org.apache.spark.sql Row]
   [scala.collection.mutable WrappedArray]))

(defn vector->clojure
  "Converts Spark vector into Clojure vector."
  [value]
  (vec (.toArray ^Vector value)))

(defn wrapped-array->clojure
  "Converts Scala WrappedArray into Clojure vector."
  [wrapped-array]
  (vec (.array ^WrappedArray wrapped-array)))

(defn normalize-value
  "Normalizes Spark-specific values into Clojure values."
  [value]
  (cond
    (instance?  Vector value)
    (vector->clojure value)

    (instance? WrappedArray value)
    (wrapped-array->clojure value)

    :else
    value))

(defn row->map
  "Converts Spark row into normalized Clojure map."
  [row fields]
  (into {}
        (map (fn [field]
               [field
                (normalize-value (.getAs ^Row row field))])
             fields)))

(defn dataset->json
  "Converts Spark dataset into JSON-safe preview."
  [ds fields]
  (mapv #(row->map % fields)
        (.collectAsList (.limit ds 20))))

(defn preview-columns
  "Combines multiple field collections into single vector."
  [& cols]
  (vec (flatten cols)))

(defn normalize-row
  "Normalizes Spark row map values."
  [row-map]
  (reduce-kv
   (fn [acc k v]
     (assoc acc k (normalize-value v)))
   {}
   row-map))

(comment
  (defn dataset->preview
    "Converts Spark dataset preview into JSON-safe maps."
    [ds]
    (->> (.limit ds 10)
         sql/collect
         (map sql/row->map)
         (map normalize-row))))