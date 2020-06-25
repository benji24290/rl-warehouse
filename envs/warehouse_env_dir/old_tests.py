# Test 1 heur vs q vs rand
        # plt.plot(random_actions(5000).all_episode_rewards_per_step, label='random')
        # plt.plot(q_function(5000, 1000, 0.1).all_episode_rewards_per_step, label='Q5k-1k-g1')
        # plt.plot(q_function(5000, 1000, 0.1, 0.5).all_episode_rewards_per_step, label='Q5k-1k-g0.5')
        # plt.plot(heuristic(5000).all_episode_rewards_per_step, label='heur-true')
        # plt.plot(heuristic(5000, False).all_episode_rewards_per_step, label='heur-false')

        # Test 2 Q different step count / different episodes
        # plt.plot(q_function(5000, 1000).all_episode_rewards_per_step, label='Q5k-1k')
        # plt.plot(q_function(10000).all_episode_rewards_per_step, label='Q10k-1k')
        # plt.plot(q_function(5000, 2000).all_episode_rewards_per_step, label='Q5k-2k')

        # Test 3 Alphas
        # plt.plot(q_function(5000, 1000, 0.1).all_episode_rewards_per_step, label='Q5k-1k-a0.1')
        # plt.plot(q_function(5000, 1000, 0.25).all_episode_rewards_per_step, label='Q5k-1k-a0.25')
        # plt.plot(q_function(5000, 1000, 0.5).all_episode_rewards_per_step, label='Q5k-1k-a0.5')

        # Test 4 Gammas
        # plt.plot(q_function(5000, 1000, 0.1, 1).all_episode_rewards_per_step, label='Q5k-1k-g1')
        # plt.plot(q_function(5000, 1000, 0.1, 0.75).all_episode_rewards_per_step, label='Q5k-1k-g0.75')
        # plt.plot(q_function(5000, 1000, 0.1, 0.5).all_episode_rewards_per_step, label='Q5k-1k-g0.5')

        # Test 5 Epsilon
        # plt.plot(q_function(5000, 1000, 0.1, 1, 1).all_episode_rewards_per_step, label='Q5k-1k-e1')
        # plt.plot(q_function(5000, 1000, 0.1, 1, 0.75).all_episode_rewards_per_step, label='Q5k-1k-e0.75')
        # plt.plot(q_function(5000, 1000, 0.1, 1, 0.5).all_episode_rewards_per_step, label='Q5k-1k-e0.5')

        # Test 6 Gamma with more steps/episodes

        # Test 7 Epsilon with more steps/episodes

        # Test 8 normal q vs q with extended
        # plt.plot(q_function(15000, 2000).all_episode_rewards_per_step, label='Q5k-1k-g1')
        # plt.plot(q_function_extended_order(15000, 2000).all_episode_rewards_per_step,
        # label='Q5k-1k-g1-extended-order')

        # Test 9
        # q_function(5000)
        # q_function_extended_order(5000)
        # heuristic(5000, False)

        # plt.plot(smoothList(random_actions(5000).all_episode_rewards_per_step,
        # degree = 400), label='random')
        # plt.plot(smoothList(q_function(5000).all_episode_rewards_per_step, degree=400), label='Q5k-1k-g1')

        # Test Summary 1 One-Art---4Steps to request

        if(False):
            rew_q_e_order = q_function_extended_order(
                300000, 100, 0.1,  seed=1234, simple_state=False, steps_to_request=4)
            rew_q_e_order_g9 = q_function_extended_order(
                300000, 100, 0.1, gamma=0.9,  seed=1234, simple_state=False, steps_to_request=4)
            rew_q_e_order_g8 = q_function_extended_order(
                300000, 100, 0.1, gamma=0.8,  seed=1234, simple_state=False, steps_to_request=4)
            rew_q_e_order_g7 = q_function_extended_order(
                300000, 100, 0.1, gamma=0.7,  seed=1234, simple_state=False, steps_to_request=4)
            rew_q_e_order_g6 = q_function_extended_order(
                300000, 100, 0.1, gamma=0.6,  seed=1234, simple_state=False, steps_to_request=4)
            rew_h_v3 = heuristic(1000, 100, version='v3',
                                 seed=1234,  steps_to_request=4)
            rew_h_v4 = heuristic(1000, 100, version='v4',
                                 seed=1234,  steps_to_request=4)
            plt.xlabel('Epochen')
            plt.ylabel('∅-Reward pro Step')
            plt.title('Bestellung alle 4 Steps')
            plt.plot(rew_q_e_order.get_smooth_all_episode_rewards_per_step(),
                     label='q-func-v2-g10')
            plt.plot(rew_q_e_order_g9.get_smooth_all_episode_rewards_per_step(),
                     label='q-func-v2-g9')
            plt.plot(rew_q_e_order_g8.get_smooth_all_episode_rewards_per_step(),
                     label='q-func-v2-g8')
            plt.plot(rew_q_e_order_g7.get_smooth_all_episode_rewards_per_step(),
                     label='q-func-v2-g7')
            plt.plot(rew_q_e_order_g6.get_smooth_all_episode_rewards_per_step(),
                     label='q-func-v2-g6')
            plt.plot(
                rew_h_v3.get_smooth_all_episode_rewards_per_step(), label='heur-v3')
            plt.plot(
                rew_h_v4.get_smooth_all_episode_rewards_per_step(), label='heur-v4')
            plt.legend()
            plt.show()

        # Test 12 one art get best alpha -> validate with 100 seeds
        if(False):

            plt.plot(q_function_with_idle(100, 1000, 1, seed=1234).all_episode_rewards_per_step,
                     label='q-a1')
            plt.plot(q_function_with_idle(100, 1000, 0.9, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.9')
            plt.plot(q_function_with_idle(100, 1000, 0.8, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.8')
            plt.plot(q_function_with_idle(100, 1000, 0.7, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.7')
            plt.plot(q_function_with_idle(100, 1000, 0.6, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.6')
            plt.plot(q_function_with_idle(100, 1000, 0.5, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.5')
            plt.plot(q_function_with_idle(100, 1000, 0.4, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.4')
            plt.plot(q_function_with_idle(100, 1000, 0.3, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.3')
            plt.plot(q_function_with_idle(100, 1000, 0.2, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.2')
            plt.plot(q_function_with_idle(100, 1000, 0.1, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.1')

            a1 = []
            a2 = []
            a3 = []
            a4 = []
            a5 = []
            a6 = []
            a7 = []
            a8 = []
            a9 = []
            a10 = []

            for i in range(100):
                a1.append(np.mean(q_function_with_idle(
                    100, 1000, 0.1, seed=i**2).all_episode_rewards_per_step))
                a2.append(np.mean(q_function_with_idle(
                    100, 1000, 0.2, seed=i**2).all_episode_rewards_per_step))
                a3.append(np.mean(q_function_with_idle(
                    100, 1000, 0.3, seed=i**2).all_episode_rewards_per_step))
                a4.append(np.mean(q_function_with_idle(
                    100, 1000, 0.4, seed=i**2).all_episode_rewards_per_step))
                a5.append(np.mean(q_function_with_idle(
                    100, 1000, 0.5, seed=i**2).all_episode_rewards_per_step))
                a6.append(np.mean(q_function_with_idle(
                    100, 1000, 0.6, seed=i**2).all_episode_rewards_per_step))
                a7.append(np.mean(q_function_with_idle(
                    100, 1000, 0.7, seed=i**2).all_episode_rewards_per_step))
                a8.append(np.mean(q_function_with_idle(
                    100, 1000, 0.8, seed=i**2).all_episode_rewards_per_step))
                a9.append(np.mean(q_function_with_idle(
                    100, 1000, 0.9, seed=i**2).all_episode_rewards_per_step))
                a10.append(np.mean(q_function_with_idle(
                    100, 1000, 1, seed=i**2).all_episode_rewards_per_step))

            plt.plot(a1, label='q-a1')
            plt.plot(a2, label='q-a2')
            plt.plot(a3, label='q-a3')
            plt.plot(a4, label='q-a4')
            plt.plot(a5, label='q-a5')
            plt.plot(a6, label='q-a6')
            plt.plot(a7, label='q-a7')
            plt.plot(a8, label='q-a8')
            plt.plot(a9, label='q-a9')
            plt.plot(a10, label='q-a10')
            print('Mean a1:', np.mean(a1))
            print('Mean a2:', np.mean(a2))
            print('Mean a3:', np.mean(a3))
            print('Mean a4:', np.mean(a4))
            print('Mean a5:', np.mean(a5))
            print('Mean a6:', np.mean(a6))
            print('Mean a7:', np.mean(a7))
            print('Mean a8:', np.mean(a8))
            print('Mean a8:', np.mean(a9))
            print('Mean a10:', np.mean(a10))
            plt.legend()
            plt.show()
        # test_prob()
        # q_function(5000)
        # heuristic(5000, False)


# Test all 10
   if(False):
        plt.plot(
            heuristic(5000, version='v1').all_episode_rewards_per_step, label='heur-v1')
        plt.plot(
            heuristic(5000, version='v2').all_episode_rewards_per_step, label='heur-v2')
        plt.plot(
            heuristic(5000, version='v3').all_episode_rewards_per_step, label='heur-v3')
        plt.plot(
            heuristic(5000, version='v4').all_episode_rewards_per_step, label='heur-v4')

        plt.plot(q_function(
            5000).all_episode_rewards_per_step, label='q')
        # plt.plot(smoothList(q_function_extended_order(5000),
        #                    degree=400), label='q-extended')
        plt.plot(q_function_with_idle(
            5000).all_episode_rewards_per_step, label='q-idle')

    # Test 11 only one ART!
    if(False):

        rew_q_w_i_a0_1 = q_function_with_idle(10000, 1000, 0.1,  seed=1234)
        rew_q_e = q_function_extended_order(10000, 1000, 0.1,  seed=1234)
        rew_h_v1 = heuristic(10000, version='v1', seed=1234)
        rew_h_v2 = heuristic(10000, version='v2', seed=1234)
        rew_h_v3 = heuristic(10000, version='v3', seed=1234)
        rew_h_v4 = heuristic(10000, version='v4', seed=1234)

        plt.plot(rew_q_w_i_a0_1.all_episode_rewards_per_step,
                 label='q-idle-a0.1')
        plt.plot(rew_q_e.all_episode_rewards_per_step,
                 label='q-ext')
        plt.plot(
            rew_h_v1.all_episode_rewards_per_step, label='h-v1')
        plt.plot(
            rew_h_v2.all_episode_rewards_per_step, label='h-v2')
        plt.plot(
            rew_h_v3.all_episode_rewards_per_step, label='h-v3')
        plt.plot(
            rew_h_v4.all_episode_rewards_per_step, label='h-v4')

        plt.legend()
        plt.show()

        rew_q_w_i_a0_1.plot_pos_neg_rewards(name='q-idle-a0.1')
        rew_q_e.plot_pos_neg_rewards(name='q-idle-a0.1')
        rew_q_w_i_a0_1.plot_episode_rewards(999)
        rew_q_e.plot_episode_rewards(999)
        rew_h_v1.plot_episode_rewards(999)
        rew_h_v2.plot_episode_rewards(999)
        rew_h_v3.plot_episode_rewards(999)
        rew_h_v4.plot_episode_rewards(999)

    # Test 11-2 only one ART!
    if(False):

        rew_q_w_i_a0_1 = q_function_with_idle(
            10000, 1000, 0.1,  seed=1234)
        rew_q_e = q_function_extended_order(10000, 1000, 0.1,  seed=1234)
        rew_h_v4 = heuristic(10000, version='v4', seed=1234)
        rew_h_v3 = heuristic(10000, version='v3', seed=1234)

        plt.plot(rew_q_w_i_a0_1.all_episode_rewards_per_step,
                 label='q-idle-a0.1')
        plt.plot(rew_q_e.all_episode_rewards_per_step,
                 label='q-ext')
        plt.plot(
            rew_h_v3.all_episode_rewards_per_step, label='h-v3')
        plt.plot(
            rew_h_v4.all_episode_rewards_per_step, label='h-v4')

        plt.legend()
        plt.show()

        rew_q_w_i_a0_1.plot_pos_neg_rewards(name='q-idle-a0.1')
        rew_q_e.plot_pos_neg_rewards(name='q-idle-a0.1')
        rew_q_w_i_a0_1.plot_episode_rewards(999)
        rew_q_e.plot_episode_rewards(999)
        rew_h_v3.plot_episode_rewards(999)
        rew_h_v4.plot_episode_rewards(999)
