import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_dim,
        tech_indicator_list,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=2,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        initial_buy=False,  # Use half of initial amount to buy
        hundred_each_trade=True,  # The number of shares per lot must be an integer multiple of 100
    ):
        self.day = day
        self.df = df
        self.date_range = df.index.levels[0].to_list() #index 0 is time

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_dim = action_dim
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.date_range[self.day]]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self._multi_stock = True if len(self.df.index.get_level_values(1).unique()) > 1 else False
        self.initial_buy = initial_buy
        self.hundred_each_trade = hundred_each_trade
        self.state = self._initiate_state()
        self.end_date = len(self.date_range) - 1

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.portfolio_memory = []
        self.actions_memory = []
        self.date_memory = []

        #self._seed()

    def _trade_stock(self, actions: np.ndarray):
        if np.any(self.state[1: self.stock_dim+1] == 0):
            raise ValueError("there are prices as zero which means the data was corrupted")
        # Sell process
        # first make sure the corresponding stock position
        current_holdings = self.state[self.stock_dim+1: self.stock_dim*2+1].copy()
        close_prices = self.state[1: self.stock_dim+1].copy()

        sell_actions = np.where(actions<0, -actions, 0)
        sell_num_shares = np.where(
            sell_actions-current_holdings<0,
            sell_actions,
            current_holdings)
        sell_amount = np.dot(close_prices, sell_num_shares)
        sell_cost = sell_amount * self.sell_cost_pct
        self.state[0] += sell_amount - sell_cost
        self.state[self.stock_dim+1: 2*self.stock_dim+1] -= sell_num_shares
        self.cost += sell_cost
        self.trades += (sell_num_shares!=0).sum()
        
        # buy process
        # Here we use the percentage of each stock to buy instead of starving others in the original code
        buy_actions = np.where(actions>0, actions, 0)
        buy_allocation = self.state[0] * buy_actions * close_prices/ np.sum(buy_actions * close_prices)
        buy_num_shares = buy_allocation // (close_prices * (1 + self.buy_cost_pct))
        # buy_num_shares = np.minimum(buy_num_shares, buy_actions)
        buy_amount = np.dot(close_prices, buy_num_shares)
        buy_cost = buy_amount * self.buy_cost_pct
        self.state[0] -= buy_amount + buy_cost
        self.state[self.stock_dim+1: 2*self.stock_dim+1] += buy_num_shares
        self.cost += buy_cost
        self.trades += (buy_num_shares!=0).sum()
        
        '''

        argsort_actions = np.argsort(actions)
        buy_indices = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
        mask = np.ones_like(actions, dtype=bool)
        mask[buy_indices] = False
        buy_act = actions.copy()
        buy_act[mask] = 0
        buy_allocation = buy_act / np.sum(buy_act[buy_indices]) * state_0 # NOTE: for not starving others
        # buy_num_shares = np.zeros(stock_dim)
        buy_num_shares = buy_allocation // (close_price * (1 + self.buy_cost_pct))
        buy_amount = np.dot(self.state[1: self.stock_dim+1], buy_num_shares)
        self.cost = buy_amount * self.buy_cost_pct
        self.state[0] -= buy_amount + self.cost
        self.state[self.stock_dim+1: 2*self.stock_dim+1] += buy_num_shares

        buy_percentage = actions / np.sum(actions[buy_indices]) # NOTE: for not starving others
        buy_num_shares = np.zeros(self.stock_dim)
        # TODO: Maybe we dont need to iterate all buy_indices
        for index in buy_indices:
            if self.state[0] > 0:
                available_amount = self.state[0] * buy_percentage[index] // (
                    self.state[index + 1] * 
                    (1 + self.buy_cost_pct)
                    )
                buy_num_shares[index] = min(available_amount, actions[index]) # ignore
                buy_amount = self.state[index + 1] * buy_num_shares[index]
                cost_amount = buy_amount * self.buy_cost_pct
                self.state[0] -= buy_amount + cost_amount
                self.cost += cost_amount       
        self.state[self.stock_dim+1: 2*self.stock_dim+1] += buy_num_shares
        self.trades += (buy_num_shares!=0).sum()
        
     
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            sell_num_shares = self.state[self.stock_dim+1: 2*self.stock_dim+1]
            sell_amount = np.dot(self.state[1: self.stock_dim+1], sell_num_shares)
            cost_amount = sell_amount * self.sell_cost_pct
            self.state[0] += sell_amount - cost_amount
            self.state[self.stock_dim: 2*self.stock_dim+1] = 0
            self.cost += cost_amount
            self.trades += self.stock_dim
        '''
        # final shares
        return buy_num_shares - sell_num_shares

    def _make_plot(self):
        portfolio_df = self._get_portfolio_df()
        plt.plot(portfolio_df["date"], portfolio_df["total_asset"], color="r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, action):
        # action is a np.ndarray with ndim as stock dimension
        # here something wrong with terminal state 
        # update next state
        if self.day >= self.end_date - 1:
            self.terminal = True  

        action = action * self.hmax
        action = action.astype(int)

        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            action = np.array([-self.hmax] * self.stock_dim)

        # calculate information before trading
        begin_cash = self.state[0]
        begin_market_value = self._get_market_value()
        begin_total_asset = begin_cash + begin_market_value
        begin_cost = self.cost
        begin_trades = self.trades
        begin_stock = self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]

        # affect state[0], cost, trade and action itself
        final_action = self._trade_stock(action)
        self.actions_memory.append(final_action) 

        self.day += 1
        self.data = self.df.loc[self.date_range[self.day]]

        self.state = self._update_state()  


        if self.turbulence_threshold is not None:
                self.turbulence = self.data["turbulence"].values[0]
        
        # calculate information after trading
        end_cash = self.state[0]
        end_market_value = self._get_market_value()
        end_total_asset = end_cash + end_market_value
        end_cost = self.cost
        end_trades = self.trades
        end_stock = self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]

        self.reward = end_total_asset - begin_total_asset
        self.reward = self.reward * self.reward_scaling

        # some panelty?
        no_trade_indices = np.where((begin_stock - end_stock) == 0)[0]
        penalty  = np.dot(self.state[no_trade_indices+1], self.state[no_trade_indices+self.stock_dim+1]) \
            * 0.001
        self.reward -= penalty

        date = self._get_date()

        step_info = {}
        step_info.update(
            {
                "date": date,
                "prev_total_asset": begin_total_asset,
                "prev_cash": begin_cash,
                "prev_market_value": begin_market_value,
                "total_asset": end_total_asset,
                "cash": end_cash,
                "market_value": end_market_value,
                "cost": end_cost - begin_cost,
                "trades": end_trades - begin_trades,
                "reward": self.reward,
            }
        ) 
        self.portfolio_memory.append(step_info)
        self.date_memory.append(date)

        self.reward = self.reward * self.reward_scaling
          
        # update next state# state makes 
        if self.terminal:
            print(f"Episode: {self.episode} end.")
            
            portfolio_df = self._get_portfolio_df()
            begin_total_asset = portfolio_df["prev_total_asset"].iloc[0]
            end_total_asset = portfolio_df["total_asset"].iloc[-1]
            tot_reward = end_total_asset - begin_total_asset

            portfolio_df["daily_return"] = portfolio_df["total_asset"].pct_change(1)

            sharpe = None
            if portfolio_df["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * portfolio_df["daily_return"].mean()
                    / portfolio_df["daily_return"].std()
                )

            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {begin_total_asset:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if sharpe is not None:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")


            trade_actions_df = self._get_actions_df()
            info = {"portfolio_df": portfolio_df, "trade_actions_df": trade_actions_df}
        else:
            info = {}

        return self.state, self.reward, self.terminal, False, info

    def reset(self, seed=None, options=None):
        # initiate state
        super().reset(seed=seed)
        self.day = 0
        # self.data = self.df.loc[self.day]
        self.data = self.df.loc[self.date_range[self.day]] # go back to the initial day

        self.state = self._initiate_state()
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.actions_memory = []
        self.date_memory = []
        self.portfolio_memory = []

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        initial_amount = self.initial_amount if self.initial else self.previous_state[0]
        close_value = self.data.close.values if self._multi_stock else [self.data.close]
        tech_value = [self.data[tech].values for tech in self.tech_indicator_list] \
            if self._multi_stock else [self.data[tech] for tech in self.tech_indicator_list]
                        # for multiple stock
        state = np.concatenate(
            ([initial_amount],close_value,[0] * self.stock_dim, tech_value), axis = None
        )
        if self.initial_buy:
            state = self._initial_buy()
        
        return state

    def _update_state(self):
        close_value = self.data.close.values if self._multi_stock else [self.data.close]
        tech_value = [self.data[tech].values for tech in self.tech_indicator_list] \
            if self._multi_stock else [self.data[tech] for tech in self.tech_indicator_list]
        
        state = np.concatenate(
            (
                [self.state[0]],
                close_value,
                self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)],
                tech_value
            ), axis = None
        )

        return state

    def _get_date(self):
        #if self._multi_stock:
            #date = self.data.date.unique()[0]
            # date = self.data.time.unique()[0] 
        #else:
            # date = self.data.date
            #date = self.data.time
        date = self.date_range[self.day]
        return date

    def _get_portfolio_df(self):
        portfolio_df = pd.DataFrame(self.portfolio_memory)
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        #portfolio_df.sort_values("date", inplace=True)
        return portfolio_df
        
    def _get_actions_df(self):
        if self._multi_stock:
            # date and close price length must match actions length
            date_list = self.date_memory
            '''
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]
            '''
            action_list = np.vstack(self.actions_memory)
            '''
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
            '''
            df_actions = pd.DataFrame(action_list, index=date_list, columns=self.data.index.values)
        else:
            date_list = self.date_memory
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _get_total_asset(self):
        """
        get current total asset value
        """
        return self.state[0] + self._get_market_value()

    def _get_market_value(self):
        """
        get current market value
        """
        return (
            self.state[1 : (self.stock_dim + 1)]
            * self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
            ).sum()

    def save_asset_memory(self):
        portfolio_df = self._get_portfolio_df()
        df_account_value = portfolio_df[["date", "total_asset"]].rename(
            columns={"total_asset": "account_value"}
        )
        return df_account_value

    def save_action_memory(self):
        if self._multi_stock > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions
    '''
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    '''
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        e = VecNormalize(e, norm_obs=True, norm_reward=True) 
        obs = e.reset()
        return e, obs

    def _initial_buy(self):
        """Initialize the state, already bought some"""
        prices = self.data.close.values
        # only use half of the initial amount
        market_values_each_tic = 0.5 * self.initial_amount // len(prices)
        buy_nums_each_tic = (market_values_each_tic // prices).astype(np.int64)
        if self.hundred_each_trade:
            buy_nums_each_tic = buy_nums_each_tic // 100 * 100 

        buy_amount = np.dot(prices, buy_nums_each_tic).sum()

        state = np.concatenate(
            (
                [self.initial_amount - buy_amount],
                prices,
                buy_nums_each_tic,
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list]

            ), axis = None
        )
        '''
            [self.initial_amount - buy_amount]
            + prices
            + buy_nums_each_tic
            + sum(
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                [],
            )
        )
        '''
        return state
