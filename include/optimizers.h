//
// Created by Pretorius, Christiaan on 2022-10-01.
//

#ifndef NNNN_OPTIMIZERS_H
#define NNNN_OPTIMIZERS_H

#include <basics.h>

namespace noodle {
    using namespace std;
    using namespace Eigen;

    namespace optimizers {
        struct adam {
            void clear() {
                isInitialized = false;
                currentTimestep = 0;
            }

            explicit adam(num_t learningRate, num_t beta1 = 0.9, num_t beta2 = 0.99, num_t epsilon = 1e-8) :
                    learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon),
                    isInitialized(false), currentTimestep(1) {}

            mat_t weightUpdate(const mat_t &gradWeights) {
                if (!isInitialized) {
                    firstMoment = mat_t(gradWeights.rows(), gradWeights.cols());
                    firstMoment.setZero();

                    secondMoment = mat_t(gradWeights.rows(), gradWeights.cols());
                    secondMoment.setZero();
                    isInitialized = true;
                }
                // m_t = B_1 * m_(t-1) + (1 - B_1) * g_t
                firstMoment = constant(firstMoment, beta1) * firstMoment +
                              constant(secondMoment, 1 - beta1) * gradWeights;

                // v_t = B_2 * v_(t-1) + (1 - B_2) * g_t^2
                secondMoment = (constant(secondMoment, beta2) * secondMoment +
                                constant(gradWeights, 1 - beta2)) * m_pow(gradWeights, 2);

                mat_t biasCorrectedFirstMoment = (firstMoment.array() /
                                                  constant(firstMoment,
                                                           1 - pow(beta1, currentTimestep)).array()).matrix();
                mat_t biasCorrectedSecondMoment = (secondMoment.array() / constant(secondMoment, 1 - pow(beta2,
                                                                                                         currentTimestep)).array()).matrix();

                currentTimestep++;
                // Return firstMoment  * (learning_rate) / (sqrt(secondMoment) + epsilon)
                return biasCorrectedFirstMoment * ((
                        (constant(gradWeights, learningRate).array() /
                         (m_sqrt(biasCorrectedSecondMoment) + constant(gradWeights, epsilon)).array()).matrix()
                ));
            };

        private:
            num_t learningRate; ///< The learning rate of our optimizer
            num_t beta1;        ///< Our B1 parameter (first moment decay)
            num_t beta2;        ///< Our B2 parameter (second moment decay)
            num_t epsilon;      ///< Stability factor

            bool isInitialized = false;      ///< On our first iteration, set the first and second order gradients to zero
            num_t currentTimestep = 0;  ///< Our current timestep (iteration)

            // Our exponentially decaying average of past gradients
            mat_t firstMoment;  ///< Our t term that represents the first order gradient decay
            mat_t secondMoment; ///< Our v_t term that represents the second order gradient decay
        };

        struct sgd {
            num_t learning_rate;
            size_t mini_batch_size;

            sgd(num_t learning_rate, size_t mini_batch_size) : learning_rate(learning_rate),
                                                               mini_batch_size(mini_batch_size) {

            }

            mat_t weightUpdate(const mat_t &gradWeights) {
                return learning_rate / mini_batch_size * gradWeights;
            }
        };
    }
}
#endif //NNNN_OPTIMIZERS_H
