CODE_SKETCHES = [

    # DEPTH = 1
    ({'type': 'run', 'children': [{'type': 'A_1', 'minval': '1'}]}, 0),

    # DEPTH = 2
    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'}
      ]}, 1),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1)',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]}
      ]}, 2),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'},
          {'type': 'repeat_2',
           'children': [
               {'type': 'A_4', 'minval': '1'}
           ]},
          {'type': 'A_5'}
      ]}, 3),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_4', 'minval': '1'}
           ]}
      ]}, 4),

    # DEPTH = 3
    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_2',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'}]},
          {'type': 'A_5'}]}, 5),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'}]},
          {'type': 'A_5'}]}, 6),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'}]},
          {'type': 'A_6'}]}, 7),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_1',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'}]},
      ]
      }, 8),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'}]},
      ]
      }, 9),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'}]}
      ]
      }, 10),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_2',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'},
               {'type': 'repeat_3',
                'children': [
                    {'type': 'A_5', 'minval': '1'}
                ]},
               {'type': 'A_6'}
           ]},
          {'type': 'A_7'}
      ]}, 11),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_2',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_6'}
           ]},
          {'type': 'A_7'}
      ]}, 12),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_2',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_6', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_7'}
           ]},
          {'type': 'A_8'}
      ]}, 13),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'},
               {'type': 'repeat_2',
                'children': [
                    {'type': 'A_5', 'minval': '1'}
                ]},
               {'type': 'A_6'}
           ]},
          {'type': 'A_7'}
      ]}, 14),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'},
               {'type': 'repeat_2',
                'children': [
                    {'type': 'A_6', 'minval': '1'}
                ]},
               {'type': 'A_7'}
           ]},
          {'type': 'A_8'}
      ]}, 15),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'},
               {'type': 'if_2',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_6'}
           ]},
          {'type': 'A_7'}
      ]}, 16),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_6', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_7'}
           ]},
          {'type': 'A_8'}
      ]}, 17),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_6'}
           ]},
          {'type': 'A_7'}
      ]}, 18),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'},
               {'type': 'ifelse_2',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_6', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_7', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_7'}
           ]},
          {'type': 'A_8'}
      ]}, 19),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_1',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'},
               {'type': 'repeat_2',
                'children': [
                    {'type': 'A_5', 'minval': '1'}
                ]},
               {'type': 'A_6'}
           ]}
      ]}, 20),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_1',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_6'}
           ]}
      ]}, 21),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_1',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_6', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_7'}
           ]}
      ]}, 22),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'},
               {'type': 'repeat_1',
                'children': [
                    {'type': 'A_5', 'minval': '1'}
                ]},
               {'type': 'A_6'}
           ]}
      ]}, 23),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'},
               {'type': 'repeat_1',
                'children': [
                    {'type': 'A_6', 'minval': '1'}
                ]},
               {'type': 'A_7'}
           ]}
      ]}, 24),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'},
               {'type': 'if_2',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_6'}
           ]}
      ]}, 25),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_6', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_7'}
           ]}
      ]}, 26),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_6', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_7'}
           ]}
      ]}, 27),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'},
               {'type': 'ifelse_2',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_6', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_7', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_8'}
           ]}
      ]}, 28),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'}
           ]},
          {'type': 'A_5'},
          {'type': 'repeat_2',
           'children': [
               {'type': 'A_6', 'minval': '1'}
           ]},
          {'type': 'A_7'}
      ]}, 29),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'}
           ]},
          {'type': 'A_6'},
          {'type': 'repeat_2',
           'children': [
               {'type': 'A_7', 'minval': '1'}
           ]},
          {'type': 'A_7'}
      ]}, 30),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_2',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'}
           ]},
          {'type': 'A_5'},
          {'type': 'repeat_3',
           'children': [
               {'type': 'A_6', 'minval': '1'}
           ]},
          {'type': 'A_7'}
      ]}, 31),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'}
           ]},
          {'type': 'A_5'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_6', 'minval': '1'}
           ]}
      ]}, 32),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'}
           ]},
          {'type': 'A_6'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_7', 'minval': '1'}
           ]}
      ]}, 33),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_2',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'}
           ]},
          {'type': 'A_5'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_6', 'minval': '1'}
           ]}
      ]}, 34),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'},
          {'type': 'repeat_2',
           'children': [
               {'type': 'A_4'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_6'}
           ]},
          {'type': 'A_7'}
      ]}, 35),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'},
          {'type': 'repeat_2',
           'children': [
               {'type': 'A_4'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_6', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_7'}
           ]},
          {'type': 'A_8'}
      ]}, 36),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'},
          {'type': 'repeat_2',
           'children': [
               {'type': 'A_4'},
               {'type': 'repeat_3',
                'children': [
                    {'type': 'A_5', 'minval': '1'}
                ]},
               {'type': 'A_6'}
           ]},
          {'type': 'A_7'}
      ]}, 37),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_4'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_6'}
           ]}
      ]}, 38),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_4'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_5', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_6', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_7'}
           ]}
      ]}, 39),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2', 'minval': '1'}
           ]},
          {'type': 'A_3'},
          {'type': 'repeatUntil_1',
           'children': [
               {'type': 'A_4'},
               {'type': 'repeat_2',
                'children': [
                    {'type': 'A_5', 'minval': '1'}
                ]},
               {'type': 'A_6'}
           ]}
      ]}, 40),
]

KAREL_SKETCHES = [
    ({'type': 'run', 'children': [{'type': 'A_1', 'minval': '1'}]}, 0),

    ({
         'type': 'run',
         'children': [
             {'type': 'A_1'},
             {'type': 'repeat_1',
              'children': [
                  {'type': 'A_2', 'minval': '1'}
              ]},
             {'type': 'A_3'},
         ]
     }, 1),

    ({
         'type': 'run',
         'children': [
             {'type': 'A_1'},
             {'type': 'while_1',
              'children': [
                  {'type': 'A_2', 'minval': '1'}
              ]},
             {'type': 'A_3'},
         ]
     }, 2),

    ({
         'type': 'run',
         'children': [
             {'type': 'A_1'},
             {'type': 'repeat_1',
              'children': [
                  {'type': 'A_2', 'minval': '1'}
              ]},
             {'type': 'A_3'},
             {'type': 'repeat_2',
              'children': [
                  {'type': 'A_4', 'minval': '1'}
              ]},
             {'type': 'A_5'},
         ]
     }, 3),

    ({
         'type': 'run',
         'children': [
             {'type': 'A_1'},
             {'type': 'while_1',
              'children': [
                  {'type': 'A_2', 'minval': '1'}
              ]},
             {'type': 'A_3'},
             {'type': 'while_2',
              'children': [
                  {'type': 'A_4', 'minval': '1'}
              ]},
             {'type': 'A_5'},
         ]
     }, 4),

    ({
         'type': 'run',
         'children': [
             {'type': 'A_1'},
             {'type': 'repeat_1',
              'children': [
                  {'type': 'A_2', 'minval': '1'}
              ]},
             {'type': 'A_3'},
             {'type': 'while_2',
              'children': [
                  {'type': 'A_4', 'minval': '1'}
              ]},
             {'type': 'A_5'},
         ]
     }, 5),

    ({
         'type': 'run',
         'children': [
             {'type': 'A_1'},
             {'type': 'while_1',
              'children': [
                  {'type': 'A_2', 'minval': '1'}
              ]},
             {'type': 'A_3'},
             {'type': 'repeat_2',
              'children': [
                  {'type': 'A_4', 'minval': '1'}
              ]},
             {'type': 'A_5'},
         ]
     }, 6),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'}]},
          {'type': 'A_5'}]}, 7),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'}]},
          {'type': 'A_6'}]}, 8),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'while_1',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'}]},
          {'type': 'A_5'}]}, 9),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'repeat_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_1',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'}]},
          {'type': 'A_5'}]}, 10),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'while_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'if_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_4'}]},
          {'type': 'A_5'}]}, 11),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'while_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'ifelse_1',
                'children': [
                    {'type': 'do',
                     'children': [
                         {'type': 'A_3', 'minval': '1'}
                     ]
                     },
                    {'type': 'else',
                     'children': [
                         {'type': 'A_4', 'minval': '1'}
                     ]
                     }
                ]},
               {'type': 'A_5'}]},
          {'type': 'A_6'}]}, 12),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'while_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'while_1',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'}]},
          {'type': 'A_5'}]}, 13),

    ({'type': 'run',
      'children': [
          {'type': 'A_1'},
          {'type': 'while_1',
           'children': [
               {'type': 'A_2'},
               {'type': 'repeat_1',
                'children': [
                    {'type': 'A_3', 'minval': '1'}
                ]},
               {'type': 'A_4'}]},
          {'type': 'A_5'}]}, 14),
]
