#-*- coding:utf-8 -*-

"""
Base Recommender Model.
"""

# Author: Anamaria Todor <anamaria@gameanalytics.com>



class BaseRecommender(BaseEstimator):
    """
    Base Class for Recommenders that suggest items for users.

    Should not be used directly, use derived classes instead

    Attributes
    ----------
     model:  DataModel
          Defines the data model where data is fetched.

     with_preference: bool
          Defines if the recommendations come along with the
          estimated preferences. (default= False)

    """

    def __init__(self, model, with_preference=False):
        self.model = model
        self.with_preference = with_preference

    def recommend(self, user_id, how_many, **params):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.
        how_many: int
                 Desired number of recommendations
        rescorer:  function, optional
                 Rescoring function to apply before final list of
                 recommendations.

        Returns
        ---------
        Return a list of recommended items, ordered from most strongly
        recommend to least.

        '''
        raise NotImplementedError("BaseRecommender is an abstract class.")

    def estimate_preference(self, user_id, item_id, **params):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.

        item_id: int or string
                Item for which recommendations are to be computed.

        Returns
        -------
        Return an estimated preference if the user has not expressed a
        preference for the item, or else the user's actual preference for the
        item. If a preference cannot be estimated, returns None.
        '''
        raise NotImplementedError("BaseRecommender is an abstract class.")

    def all_other_items(self, user_id, **params):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.

        Returns
        --------
        Return all items in the `model` for which the user has not expressed
        the preference and could possibly be recommended to the user.
        '''
        raise NotImplementedError("BaseRecommender is an abstract class.")

    def set_preference(self, user_id, item_id, value):
        '''
        Set a new preference of a user for a specific item with a certain
        magnitude.

        Parameters
        ----------
        user_id: int or string
                 User for which the preference will be updated.

        item_id: int or string
                 Item that will be updated.

        value:  The new magnitude for the preference of a item_id from a
                user_id.

        '''
        self.model.set_preference(user_id, item_id, value)

    def remove_preference(self, user_id, item_id):
        '''
        Remove a preference of a user for a specific item

        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.
        item_id: int or string
                 Item that will be removed the preference for the user_id.

        '''
        self.model.remove_preference(user_id, item_id)