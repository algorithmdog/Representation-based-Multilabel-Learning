#!/bin/python
import active
import unittest
import numpy as np;
import common

from Matrix_Utils import *;


class ActiveTester(unittest.TestCase):
    def test_active(self):
        a        = np.array([[1.0,2],[0,0],[-1,-2]]);
        
        # test sgmoid
        standard   = np.array([[0.7310585786300049,0.8807970779778823],[0.5,0.5], [0.2689414213699951, 0.11920292202211755]]);
        a_sgmoid   = active.active(a);
        self.assertEqual(is_matrix_equals(a_sgmoid, standard), True);
                
        # test linear
        a_linear = active.active(a, 'linear');
        self.assertEqual(is_matrix_equals(a_linear,a), True); 
        
        # test tanh
        a = np.array([[0.1,-0.2],[0,10]]);
        standard = np.array([[0.099667994624955902, -0.197375320224904],\
                             [0                   ,  0.99999999587769262]]);
        a_tanh = active.active(a, "tanh");
        self.assertTrue(is_matrix_equals(a_tanh, standard));

        # test rel
        standard = np.array([[0.1, 0],[0,10]]);
        a_rel = active.active(a, "relu");
        self.assertTrue(is_matrix_equals(a_rel, standard));

        with self.assertRaises(Exception):
            active.active(a,"unknown active_type")

    def test_loss(self):
        a   = np.array([[0.1,0.2],[0.3,0.9]]);
        y   = np.array([[1,0],[0,1]]);
        idx = np.array([[1,1],[1,0]]);

        loss = active.loss(a,y);
        self.assertLess(abs(loss-2.987764103904814), 0.000001);
        loss = active.loss(a,y,idx=idx);
        self.assertLess(abs(loss-2.8824035882469876), 0.000001);

        loss = active.loss(a,y,"least_square");
        self.assertLess(abs(loss-0.9500000000), 0.000001);        
        loss = active.loss(a,y,"least_square",idx);
        self.assertLess(abs(loss-0.9400000000), 0.000001);
        

        '''
        a = np.array([[0.1,0.2], [3,9]]);
        y = np.array([[1,0],[0,1]]);
        idx = np.array([[0,1],[1,1]]);
        loss = active.loss(a,y,"appro_l1_hinge");
        self.assertLess(abs(loss - 6.091), 0.0000001);
        loss = active.loss(a,y,"appro_l1_hinge", idx);
        self.assertLess(abs(loss-(6.091-0.891)), 0.00000001);

        
        loss = active.loss(a,y,"l2_hinge");
        self.assertLess(abs(loss - 18.25), 0.0000001);
        loss = active.loss(a,y,"l2_hinge",idx);
        self.assertLess(abs(loss - (18.25-0.81)), 0.0000001);
        '''

        with self.assertRaises(Exception):
            active.loss(a,y, "unkonw loss");
        

    def test_grad(self):
        a = np.array([[0.1, 0.2], [0.3, 0.9]]);
        y = np.array([[1,   0],   [0,   1]]);

        with self.assertRaises(Exception):
            active.grad(a, grad_type = "sgmoid_negative_log_likelihood");
            active.grad(a, grad_type = "linear_least_square");
            active.grad(a, grad_type = "linear_appro_l1_hinge");
            active.grad(a, grad_type = "linear_l2_hinge");
        
        grad = active.grad(a, y, grad_type = "sgmoid_negative_log_likelihood");
        tx   = np.array([[-0.9,0.2],[0.3,-0.1]]);
        self.assertTrue(is_matrix_equals(grad, tx), True);

        grad = active.grad(a, y, grad_type = "linear_least_square");
        tx   = np.array([[-1.8,0.4],[0.6,-0.2]]);
        self.assertTrue(is_matrix_equals(grad, tx), True);

       
        a  = np.array([[1, 1, -2,1]])
        y  = np.array([[1, 1, 0,0]])
        grad = active.grad(a, y, grad_type = common.grad.linear_weighted_approximate_rank_pairwise)
        st = np.array([[-1.0,-1.0, 0,2]])
        self.assertTrue(is_matrix_equals(grad, st), True)

 
        ## new test data for hinge function
        a = np.array([[0.1, -2],[0.3, 9]]);        
        y = np.array([[1,0],[0,1]]);

        '''
        grad = active.grad(a, y, grad_type = "linear_appro_l1_hinge");
        tx   = np.array([[-1.17,0],[1,0]]);
        self.assertTrue(is_matrix_equals(grad, tx), True);

        grad = active.grad(a, y, grad_type = "linear_l2_hinge");
        tx   = np.array([[-1.8,0],[2.6,0]]);
        self.assertTrue(is_matrix_equals(grad, tx), True);
        ''' 

        a = np.array([[0.1,0.2],[0.3,0.9]]);

        grad = active.grad(a, grad_type = "sgmoid");
        tx   = np.array([[0.09,0.16],[0.21,0.09]]);
        self.assertTrue(is_matrix_equals(grad, tx), True);

        grad = active.grad(a, grad_type = "linear");
        tx   = np.array([[1,1],[1,1]]);
        self.assertTrue(is_matrix_equals(grad, tx), True);

        grad = active.grad(a, grad_type = "tanh");
        tx   = np.array([[0.99,0.96],[0.91,0.19]]);
        self.assertTrue(is_matrix_equals(grad, tx), True);

        a = np.array([[0.1, -0.2], [-0.3, 10]]); 
       
        grad = active.grad(a, grad_type = "relu");
        tx   = np.array([[1, 0],[0,1]]);
        self.assertTrue(is_matrix_equals(grad, tx), True);

        with self.assertRaises(Exception):
            active.grad(a,y, "unknow active_type");
