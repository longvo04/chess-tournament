"""
Test script để kiểm tra tỉ lệ thắng của các agents.
Chạy nhiều ván đấu giữa các agents và báo cáo kết quả.
"""
import time
from agents import create_agent, RandomAgent, MinimaxAgent, MLAgent
from tournament import Tournament


def test_agent_vs_random(agent_type: str, num_games: int = 100):
    """
    Test một agent chống lại Random agent.
    Trả về tỉ lệ thắng của agent.
    """
    print(f"\n{'='*60}")
    print(f"Testing {agent_type} vs Random ({num_games} games)")
    print('='*60)
    
    agent = create_agent(agent_type)
    random_agent = create_agent("random")
    
    tournament = Tournament(
        agent1=agent,
        agent2=random_agent,
        num_matches=num_games,
        name=f"test_{agent_type}_vs_random"
    )
    
    start_time = time.time()
    tournament.run(callback=lambda m, s: print(f"\rMatch {m}/{num_games}...", end="") if m % 10 == 0 else None)
    elapsed = time.time() - start_time
    
    stats = tournament.stats
    
    print(f"\n\nResults after {elapsed:.1f} seconds:")
    print(f"-" * 40)
    print(f"{agent_type}:")
    print(f"  Wins:     {stats.agent1_wins}")
    print(f"  Losses:   {stats.agent1_losses}")
    print(f"  Win Rate: {stats.get_win_rate(1):.1f}%")
    print(f"\nRandom:")
    print(f"  Wins:     {stats.agent2_wins}")
    print(f"  Losses:   {stats.agent2_losses}")
    print(f"  Win Rate: {stats.get_win_rate(2):.1f}%")
    print(f"\nDraws: {stats.draws}")
    print(f"Draw Rate: {stats.get_draw_rate():.1f}%")
    
    return stats.get_win_rate(1)


def test_all_agents(num_games: int = 100):
    """Test tất cả agents và kiểm tra yêu cầu BTL"""
    print("\n" + "="*60)
    print("CHESS AI AGENT TESTING")
    print("="*60)
    print(f"\nRunning {num_games} games for each agent vs Random...")
    
    results = {}
    
    # Test Minimax
    results['Minimax'] = test_agent_vs_random("minimax", num_games)
    
    # Test ML
    results['ML'] = test_agent_vs_random("ml", num_games)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - BTL REQUIREMENTS CHECK")
    print("="*60)
    
    print(f"\n{'Agent':<15} {'Win Rate':<15} {'Required':<15} {'Status':<10}")
    print("-" * 55)
    
    minimax_ok = results['Minimax'] >= 90
    ml_ok = results['ML'] >= 60
    
    print(f"{'Minimax':<15} {results['Minimax']:.1f}%{'':<9} {'≥ 90%':<15} {'✓ PASS' if minimax_ok else '✗ FAIL':<10}")
    print(f"{'ML':<15} {results['ML']:.1f}%{'':<9} {'≥ 60%':<15} {'✓ PASS' if ml_ok else '✗ FAIL':<10}")
    
    print("\n" + "-" * 55)
    
    if minimax_ok and ml_ok:
        print("✓ ALL REQUIREMENTS MET!")
    else:
        print("✗ Some requirements not met. See details above.")
        if not minimax_ok:
            penalty = int((90 - results['Minimax']) / 10) * 2
            print(f"  - Minimax: {penalty} points deducted")
        if not ml_ok:
            penalty = int((60 - results['ML']) / 10) * 2
            print(f"  - ML: {penalty} points deducted")
    
    return results


if __name__ == "__main__":
    import sys
    
    num_games = 100
    if len(sys.argv) > 1:
        try:
            num_games = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of games: {sys.argv[1]}")
            sys.exit(1)
    
    test_all_agents(num_games)
