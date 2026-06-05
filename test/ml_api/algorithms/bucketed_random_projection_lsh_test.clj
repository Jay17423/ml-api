(ns ml-api.algorithms.bucketed-random-projection-lsh-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.bucketed-random-projection-lsh :as brp-lsh]
   [ml-api.algorithms.vector-assembler :as va]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "BucketedRandomProjectionLSH executes successfully"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)
                  brp-lsh/dataset->json (fn [_ _]
                                          [{:hashes [[1.0] [2.0]]}])
                  log/info (fn [_])
                  brp-lsh/execute (fn [_ _]
                                    {:data [{:hashes [[1.0] [2.0]]}]})]

      (let [res (brp-lsh/execute :dataset
                                 {:feature_field ["feature1" "feature2"]
                                  :output_field "hashes"
                                  :bucket_length 2.0
                                  :num_hash_tables 3
                                  :random_seed 10})]

        (is (= {:data [{:hashes [[1.0] [2.0]]}]}
               res))))))

(deftest execute-default-values
  (testing "BucketedRandomProjectionLSH uses default values"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)
                  brp-lsh/dataset->json (fn [_ output-field]
                                          (is (= "hashes" output-field))
                                          [{:hashes [[1.0]]}])
                  log/info (fn [_])
                  brp-lsh/execute (fn [_ _]
                                    {:data [{:hashes [[1.0]]}]})]

      (let [res (brp-lsh/execute :dataset
                                 {:feature_field ["feature1"]
                                  :bucket_length 1.5})]
        (is (= {:data [{:hashes [[1.0]]}]} res))))))

(deftest execute-negative-and-edge
  (testing "Feature vector creation failure throws exception"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             (throw (RuntimeException.
                                                     "vector creation failed")))
                  log/info (fn [_])
                  brp-lsh/execute
                  (fn [_ _]
                    (throw (ex-info
                            "BucketedRandomProjectionLSH execution failed"
                            {})))]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"BucketedRandomProjectionLSH execution failed"
           (brp-lsh/execute :dataset
                            {:feature_field ["feature1"]
                             :bucket_length 1.0})))))

  (testing "dataset->json failure throws exception"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  brp-lsh/dataset->json (fn [_ _]
                                          (throw (RuntimeException.
                                                  "json conversion failed")))
                  log/info (fn [_])
                  brp-lsh/execute
                  (fn [_ _]
                    (throw (ex-info
                            "BucketedRandomProjectionLSH execution failed"
                            {})))]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"BucketedRandomProjectionLSH execution failed"
           (brp-lsh/execute :dataset
                            {:feature_field ["feature1"]
                             :bucket_length 1.0}))))))
