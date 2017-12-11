//===================================
// 乱数のUtility
//===================================
#ifndef __RANDOM_UTILITY_H__
#define __RANDOM_UTILITY_H__

#include<boost/random.hpp>

#include"Common/Common.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Random
	{
	private:
		boost::random::mt19937 gen;	/**< 乱数ジェネレータ */

		boost::random::uniform_real_distribution<F32> distF;

	public:
		/** コンストラクタ */
		Random();
		/** デストラクタ */
		virtual ~Random();

	public:
		/** 初期化.
			乱数の種を固定値で指定して頭出ししたい場合に使用する. */
		void Initialize(U32 seed);

	private:

		/** 0.0 ～ 1.0の範囲で値を取得する */
		F32 GetValue();

	public:
		/** 一様乱数を取得する */
		F32 GetUniformValue(F32 min, F32 max);

		/** 正規乱数を取得する.
			@param	average	平均
			@param	sigma	標準偏差＝√分散 */
		F32 GetNormalValue(F32 average, F32 sigma);

		/** 切断正規乱数を取得する.
			@param	average	平均
			@param	sigma	標準偏差＝√分散 */
		F32 GetTruncatedNormalValue(F32 average, F32 sigma);
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif	__RANDOM_UTILITY_H__