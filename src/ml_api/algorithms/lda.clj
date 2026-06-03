(ns ml-api.algorithms.lda
  "Spark LDA implementation."
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.count-vectorizer :as cv]
   [ml-api.algorithms.stop-words-remover :as swr]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.clustering LDA]
   [scala.collection JavaConverters]))

(defn scala-seq->vec
  "Converts Scala sequence into Clojure vector."
  [scala-seq]
  (vec (iterator-seq (.iterator (JavaConverters/asJavaIterable scala-seq)))))

(defn row->clojure
  "Converts transformed dataset row into Clojure map."
  [row text-field topic-distribution-field]
  {:input_text (.getAs row text-field)
   :topic_distribution (utils/vector->clojure
                        (.getAs row topic-distribution-field))})

(defn dataset->json
  "Converts transformed dataset into JSON preview."
  [dataset text-field topic-distribution-field]
  (mapv #(row->clojure % text-field topic-distribution-field)
        (.collectAsList dataset)))

(defn topic-row->clojure
  "Converts topic row into readable topic keywords."
  [row vocabulary]
  (let [indices (scala-seq->vec (.getAs row "termIndices"))
        weights (scala-seq->vec (.getAs row "termWeights"))]
    {:topic (.getAs row "topic")
     :words (mapv #(nth vocabulary (int %)) indices)
     :term_indices (mapv int indices)
     :term_weights weights}))

(defn topics->json
  "Converts topics dataset into JSON."
  [topics vocabulary]
  (mapv #(topic-row->clojure % vocabulary) (.collectAsList topics)))

(defn execute
  "Executes Spark LDA."
  [dataset {:keys [text_field
                   topic_count
                   max_iterations
                   random_seed
                   training_method
                   checkpoint_interval
                   initial_learning_offset
                   learning_decay_rate
                   training_sample_rate
                   auto_optimize
                   document_concentration
                   topic_concentration
                   topic_distribution_field
                   keep_last_checkpoint]
            :or {topic_count 10
                 max_iterations 20
                 training_method "online"
                 checkpoint_interval 10
                 initial_learning_offset 1024.0
                 learning_decay_rate 0.51
                 training_sample_rate 0.05
                 auto_optimize true
                 topic_distribution_field "topicDistribution"
                 keep_last_checkpoint true}}]

  (try
    (log/info {:msg "Starting LDA"
               :text-field text_field
               :topic-count topic_count
               :training-method training_method})

    (let [filtered-dataset (swr/transform dataset text_field "filtered_words"
                                          nil
                                          false)
          cv-model-data (cv/fit-transform filtered-dataset "filtered_words"
                                          "features"
                                          262144
                                          1
                                          1.0)
          vectorized-dataset (:dataset cv-model-data)
          cv-model (:model cv-model-data)
          vocabulary (vec (.vocabulary cv-model))
          lda (-> (LDA.)
                  (.setK topic_count)
                  (.setMaxIter max_iterations)
                  (.setFeaturesCol "features")
                  (.setTopicDistributionCol topic_distribution_field)
                  (.setOptimizer training_method)
                  (.setCheckpointInterval checkpoint_interval)
                  (.setLearningOffset (double initial_learning_offset))
                  (.setLearningDecay (double learning_decay_rate))
                  (.setSubsamplingRate (double training_sample_rate))
                  (.setOptimizeDocConcentration auto_optimize)
                  (.setKeepLastCheckpoint keep_last_checkpoint))

          lda (if random_seed
                (.setSeed lda random_seed)
                lda)
          lda (if document_concentration
                (.setDocConcentration lda (double-array document_concentration))
                lda)
          lda (if topic_concentration
                (.setTopicConcentration lda (double topic_concentration))
                lda)
          model (.fit lda vectorized-dataset)
          transformed-dataset (.transform model vectorized-dataset)
          topics (.describeTopics model)
          preview (dataset->json transformed-dataset
                                 text_field
                                 topic_distribution_field)
          topic-preview (topics->json topics vocabulary)]
      (log/info {:msg "LDA completed successfully"})
      {:data preview
       :topics topic-preview})
    (catch Exception err
      (throw (ex-info "LDA execution failed"
                      {:type :algorithm/lda-failed
                       :text-field text_field
                       :error (.getMessage err)}
                      err)))))