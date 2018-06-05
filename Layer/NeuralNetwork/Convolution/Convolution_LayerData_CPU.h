//======================================
// 畳み込みニューラルネットワークのレイヤーデータ
//======================================
#ifndef __CONVOLUTION_LAYERDATA_CPU_H__
#define __CONVOLUTION_LAYERDATA_CPU_H__

#include"Convolution_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Convolution_LayerData_CPU : public Convolution_LayerData_Base
	{
		friend class Convolution_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Convolution_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Convolution_LayerData_CPU();


		//===========================
		// 初期化
		//===========================
	public:
		using Convolution_LayerData_Base::Initialize;

		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		ErrorCode Initialize(void);


		//===========================
		// レイヤー保存
		//===========================
	public:

		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);


		//===========================
		// オプティマイザー設定
		//===========================
	public:
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif