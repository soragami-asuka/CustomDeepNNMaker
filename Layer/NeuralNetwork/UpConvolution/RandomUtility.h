//===================================
// 乱数のUtility
//===================================
#pragma once

#ifdef _CRTDBG_MAP_ALLOC
#undef new
#endif

#include<boost/random.hpp>

#ifdef _CRTDBG_MAP_ALLOC
#define new  ::new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif


namespace Utility
{
	class Random
	{
	private:
		boost::random::mt19937 gen;	/**< 乱数ジェネレータ */

		boost::random::uniform_int_distribution<> distI16;
		boost::random::uniform_int_distribution<> distI8;
		boost::random::uniform_int_distribution<> distI1;
		boost::random::uniform_real_distribution<> distF;

	private:
		/** コンストラクタ */
		Random();
		/** デストラクタ */
		virtual ~Random();

	private:
		/** インスタンスの取得 */
		static Random& GetInstance();

	public:
		/** 初期化.
			乱数の種を固定値で指定して頭出ししたい場合に使用する. */
		static void Initialize(unsigned __int32 seed);

	public:
		/** 0～65535の範囲で値を取得する */
		static unsigned __int16 GetValueShort();

		/** 0～255の範囲で値を取得する */
		static unsigned __int8 GetValueBYTE();

		/** 0or1の範囲で値を取得する */
		static bool GetValueBIT();

		/** 0.0 ～ 1.0の範囲で値を取得する */
		static double GetValue();

		/** min ～ maxの範囲で値を取得する */
		static double GetValue(double min, double max);
	};
}
