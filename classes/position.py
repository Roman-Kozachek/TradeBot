class Position:
    
    def __init__(self, state, direction, exchange_rate, transaction_cost, cost_usd, usd_amount, eur_amount, leverage):
        self.state = state  # State at which the position was created
        self.direction = direction  # Direction, Ask(1) or Bid(0)
        
        self.open_date = state.Date
        self.open_exchange_rate = exchange_rate
        self.open = True
        
        self.close_date = None
        self.close_exchange_rate = 0
        
        self.transaction_cost = transaction_cost
        
        self.cost_usd = cost_usd  # Cost of the trade
        self.usd_amount = usd_amount  # Leveraged base currency amount
        self.eur_amount = eur_amount  # Leveraged target currency amount
        self._leverage = leverage
        self.current_profit = 0
        
        
    def _update_position(self, current_state):
        exchange_rate = current_state.Bid if self.direction==1 else current_state.Ask
        cur_usd_amount = self.eur_amount*exchange_rate
        self.current_profit = (cur_usd_amount - self.usd_amount) if self.direction==1 else (self.usd_amount - cur_usd_amount)
    
    
    def get_value(self):
        """
        Returns current value of an open trade and calculates profit
        """
        return round(self.cost_usd + self.current_profit - self.transaction_cost, 2)
    
    def get_profit(self):
        """
        Returns current profit of an open trade
        """
        return round(self.current_profit - self.transaction_cost*2, 2)
    
    def close(self, current_state):
        """
        Closes an open position
        """
        self.close_exchange_rate = current_state.Bid if self.direction==1 else current_state.Ask
        self.close_date = current_state.Date
        self.open = False
          
            
    def get_info(self):
        return {"Date": self.open_date, "Type": "Ask" if self.direction==1 else "Bid",\
                "At": self.open_exchange_rate, "Open": self.open, "C_Date": self.close_date,\
                "C_At": self.close_exchange_rate, \
                "Profit": self.current_profit-self.transaction_cost*2}