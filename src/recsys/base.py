"""
Base Recommender Model.
"""

# Author: Anamaria Todor <anamaria@gameanalytics.com>



class BaseRecommender():
    """
    Abstract Base Class for Recommenders that suggest items for users.

    Attributes
    ----------
     model:  DataModel
          Defines the data model where data is fetched.

     with_preference: bool
          Defines if the recommendations come along with the
          estimated preferences. (default= False)

    """

    def __init__(self, data, with_preference=False):
        self.data = data
        self.with_preference = with_preference

    def recommend(self, user_id, how_many):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.
        how_many: int
                 Desired number of recommendations

        Returns
        ---------
        Return a list of recommended items, ordered from most to least
        recommended.

        '''
        raise NotImplementedError("BaseRecommender is an abstract class.")

    def predict(self, user_id, item_id):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which the preference is to be computed.

        item_id: int or string
                Item for which the preference is to be computed.

        Returns
        -------
        Returns the prediction of the user rating if the user has not explicitly rated
        the item, or else the user's actual rating for the
        item. If a rating cannot be estimated, returns None.
        '''
        raise NotImplementedError("BaseRecommender is an abstract class.")