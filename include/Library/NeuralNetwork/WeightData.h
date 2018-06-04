//===============================================
// 最適化ルーチン
//===============================================
#ifndef __GRAVISBELL_LIBRARY_NN_WEIGHTDATA_H__
#define __GRAVISBELL_LIBRARY_NN_WEIGHTDATA_H__

#ifdef WEIGHTDATA_EXPORTS
#define WeightData_API __declspec(dllexport)
#else
#define WeightData_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#endif
#endif

#include"Layer/NeuralNetwork/IWeightData.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** 初期化管理クラス */
	class IWeightDataManager
	{
	public:
		/** コンストラクタ */
		IWeightDataManager(){};
		/** デストラクタ */
		virtual ~IWeightDataManager(){}

	public:
		/** 重みデータを作成する(CPU) */
		virtual IWeightData* CreateWeightData_CPU(const wchar_t i_weigthDataID[], U32 i_weightSize, U32 i_biasSize) = 0;
		/** 重みデータを作成する(GPU) */
		virtual IWeightData* CreateWeightData_GPU(const wchar_t i_weigthDataID[], U32 i_weightSize, U32 i_biasSize) = 0;
	};

	/** 初期化管理クラスを取得する */
	WeightData_API IWeightDataManager& GetWeightDataManager(void);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
