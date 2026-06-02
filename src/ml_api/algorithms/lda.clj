(ns ml-api.algorithms.lda
  (:require
   [taoensso.timbre :as log]
   [ml-api.algorithms.count-vectorizer :as cv])

  (:import
   [org.apache.spark.ml.clustering LDA]
   [org.apache.spark.ml.linalg Vector]))

;; ------------------------------------------------------------
;; Helpers
;; ------------------------------------------------------------

(defn vector->clojure
  [v]

  (when v
    (vec
     (.toArray ^Vector v))))

(defn row->clojure
  [row topic-distribution-field]

  {(keyword topic-distribution-field)

   (vector->clojure
    (.getAs row topic-distribution-field))})

(defn dataset->json
  [dataset topic-distribution-field]

  (mapv

   #(row->clojure
     %
     topic-distribution-field)

   (.collectAsList dataset)))

;; ------------------------------------------------------------
;; Main Execution
;; ------------------------------------------------------------

(defn execute
  "Executes Spark LDA."

  [dataset
   {:keys [text_field
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

    ;; --------------------------------------------------------
    ;; Logging
    ;; --------------------------------------------------------

    (log/info

     {:msg "Starting LDA"
      :text-field text_field
      :topic-count topic_count
      :training-method training_method})

    ;; --------------------------------------------------------
    ;; CountVectorizer Pipeline
    ;; --------------------------------------------------------

    (let [vectorized-dataset

          (cv/transform
           dataset
           text_field
           "features"
           262144
           1
           1.0)

          ;; --------------------------------------------------
          ;; Create LDA
          ;; --------------------------------------------------

          lda

          (-> (LDA.)
              (.setK topic_count)
              (.setMaxIter max_iterations)
              (.setFeaturesCol "features")
              (.setTopicDistributionCol
               topic_distribution_field)
              (.setOptimizer training_method)
              (.setCheckpointInterval
               checkpoint_interval)
              (.setLearningOffset
               (double initial_learning_offset))
              (.setLearningDecay
               (double learning_decay_rate))
              (.setSubsamplingRate
               (double training_sample_rate))
              (.setOptimizeDocConcentration
               auto_optimize)
              (.setKeepLastCheckpoint
               keep_last_checkpoint))

          ;; --------------------------------------------------
          ;; Optional Params
          ;; --------------------------------------------------

          lda

          (if random_seed
            (.setSeed lda random_seed)
            lda)

          lda

          (if document_concentration
            (.setDocConcentration
             lda
             (double-array document_concentration))
            lda)

          lda

          (if topic_concentration
            (.setTopicConcentration
             lda
             (double topic_concentration))
            lda)

          ;; --------------------------------------------------
          ;; Fit Model
          ;; --------------------------------------------------

          model
          (.fit lda
                vectorized-dataset)

          ;; --------------------------------------------------
          ;; Transform Dataset
          ;; --------------------------------------------------

          transformed-dataset
          (.transform model
                      vectorized-dataset)

          ;; --------------------------------------------------
          ;; JSON Preview
          ;; --------------------------------------------------

          preview
          (dataset->json
           transformed-dataset
           topic_distribution_field)]

      ;; ------------------------------------------------------
      ;; Success Logging
      ;; ------------------------------------------------------

      (log/info
       {:msg "LDA completed successfully"})

      ;; ------------------------------------------------------
      ;; Response
      ;; ------------------------------------------------------

      {:algorithm "LDA"

       :topics
       topic_count

       :data
       preview})

    ;; --------------------------------------------------------
    ;; Error Handling
    ;; --------------------------------------------------------

    (catch Exception err

      (throw
       (ex-info
        "LDA execution failed"

        {:type :algorithm/lda-failed
         :text-field text_field
         :error (.getMessage err)}

        err)))))