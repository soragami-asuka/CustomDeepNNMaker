//===============================================
// 最適化ルーチン
//===============================================
#ifndef __GRAVISBELL_LIBRARY_NN_INITIALIZER_H__
#define __GRAVISBELL_LIBRARY_NN_INITIALIZER_H__

#ifdef Initializer_EXPORTS
#define Initializer_API __declspec(dllexport)
#else
#define Initializer_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#endif
#endif

#include"Layer/NeuralNetwork/IInitializer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** 初期化管理クラス */
	class IInitializerManager
	{
	public:
		/** コンストラクタ */
		IInitializerManager(){};
		/** デストラクタ */
		virtual ~IInitializerManager(){}

	public:
		/** 乱数を初期化する */
		virtual ErrorCode InitializeRandomParameter() = 0;
		/** 乱数を初期化する */
		virtual ErrorCode InitializeRandomParameter(U32 i_seed) = 0;

		/** 初期化クラスを取得する */
		virtual IInitializer& GetInitializer(const wchar_t i_initializerID[]) = 0;
	};

	/** 初期化管理クラスを取得する */
	Initializer_API IInitializerManager& GetInitializerManager(void);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
