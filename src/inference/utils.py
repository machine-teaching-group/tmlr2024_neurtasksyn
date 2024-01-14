from src.symexecution.symworld import SymWorld

sk_dict = {
    "hoc": {
        "spec0": {'type': 'run',
                  'children': [
                      {'type': 'A_1'},
                      {'type': 'repeat_1',
                       'children': [
                           {'type': 'A_2'},
                       ]
                       },
                      {'type': 'A_3'}
                  ]
                  },
        "spec1": {'type': 'run',
                  'children': [
                      {'type': 'A_1'},
                      {'type': 'repeatUntil_1',
                       'children': [
                           {'type': 'A_2'},
                       ]
                       },
                  ]
                  },
        "spec2": {'type': 'run',
                  'children': [
                      {'type': 'A_1'},
                      {'type': 'repeat_1',
                       'children': [
                           {'type': 'A_2'},
                       ]
                       },
                      {'type': 'A_3'},
                      {'type': 'repeat_2',
                       'children': [
                           {'type': 'A_4'},
                       ]
                       },
                  ]
                  },
        "spec3": {'type': 'run',
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
                           {'type': 'A_5'}]},
                  ]
                  },
        "spec4": {'type': 'run',
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
                           {'type': 'A_6'}]},
                  ]
                  }
    },
    "karel": {
        "spec5": {'type': 'run',
                  'children': [
                      {'type': 'A_1'}
                  ]
                  },
        "spec6": {'type': 'run',
                  'children': [
                      {'type': 'A_1'},
                      {'type': 'while_1',
                       'children': [
                           {'type': 'A_2'},
                       ]
                       },
                      {'type': 'A_3'}
                  ]
                  },
        "spec7": {'type': 'run',
                  'children': [
                      {'type': 'A_1'},
                      {'type': 'while_1',
                       'children': [
                           {'type': 'A_2'},
                       ]
                       },
                      {'type': 'A_3'},
                      {'type': 'while_2',
                       'children': [
                           {'type': 'A_4'},
                       ]
                       },
                      {'type': 'A_5'},
                  ]
                  },
        "spec8": {'type': 'run',
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
                      {'type': 'A_5'}
                  ]
                  },
        "spec9": {'type': 'run',
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
                      {'type': 'A_5'}
                  ]
                  }
    }
}


def get_hoc_symworld():
    init_world = SymWorld.empty_init(16, 16, None)

    init_world.blocked[(3, 0)] = True
    init_world.unknown[(3, 0)] = False
    init_world.blocked[(3, 1)] = True
    init_world.unknown[(3, 1)] = False
    init_world.blocked[(3, 2)] = True
    init_world.unknown[(3, 2)] = False
    init_world.blocked[(3, 3)] = True
    init_world.unknown[(3, 3)] = False
    init_world.blocked[(2, 3)] = True
    init_world.unknown[(2, 3)] = False
    init_world.blocked[(1, 3)] = True
    init_world.unknown[(1, 3)] = False
    init_world.blocked[(0, 3)] = True
    init_world.unknown[(0, 3)] = False
    init_world.blocked[(0, 0)] = False
    init_world.unknown[(0, 0)] = False
    init_world.blocked[(0, 1)] = False
    init_world.unknown[(0, 1)] = False
    init_world.blocked[(0, 2)] = False
    init_world.unknown[(0, 2)] = False
    init_world.blocked[(1, 0)] = False
    init_world.unknown[(1, 0)] = False
    init_world.blocked[(1, 1)] = False
    init_world.unknown[(1, 1)] = False
    init_world.blocked[(1, 2)] = False
    init_world.unknown[(1, 2)] = False
    init_world.blocked[(2, 0)] = False
    init_world.unknown[(2, 0)] = False
    init_world.blocked[(2, 1)] = False
    init_world.unknown[(2, 1)] = False
    init_world.blocked[(2, 2)] = False
    init_world.unknown[(2, 2)] = False

    # do the same in the opposite corner
    init_world.blocked[(12, 15)] = True
    init_world.unknown[(12, 15)] = False
    init_world.blocked[(12, 14)] = True
    init_world.unknown[(12, 14)] = False
    init_world.blocked[(12, 13)] = True
    init_world.unknown[(12, 13)] = False
    init_world.blocked[(12, 12)] = True
    init_world.unknown[(12, 12)] = False
    init_world.blocked[(13, 12)] = True
    init_world.unknown[(13, 12)] = False
    init_world.blocked[(14, 12)] = True
    init_world.unknown[(14, 12)] = False
    init_world.blocked[(15, 12)] = True
    init_world.unknown[(15, 12)] = False
    init_world.blocked[(15, 15)] = False
    init_world.unknown[(15, 15)] = False
    init_world.blocked[(15, 14)] = False
    init_world.unknown[(15, 14)] = False
    init_world.blocked[(15, 13)] = False
    init_world.unknown[(15, 13)] = False
    init_world.blocked[(14, 15)] = False
    init_world.unknown[(14, 15)] = False
    init_world.blocked[(14, 14)] = False
    init_world.unknown[(14, 14)] = False
    init_world.blocked[(14, 13)] = False
    init_world.unknown[(14, 13)] = False
    init_world.blocked[(13, 15)] = False
    init_world.unknown[(13, 15)] = False
    init_world.blocked[(13, 14)] = False
    init_world.unknown[(13, 14)] = False
    init_world.blocked[(13, 13)] = False
    init_world.unknown[(13, 13)] = False

    return init_world