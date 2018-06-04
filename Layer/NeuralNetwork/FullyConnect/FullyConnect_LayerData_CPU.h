//======================================
// 全結合ニューラルネットワークのレイヤーデータ
//======================================
#ifndef __FullyConnect_LAYERDATA_CPU_H__
#define __FullyConnect_LAYERDATA_CPU_H__

#include"FullyConnect_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FullyConnect_LayerData_CPU : public FullyConnect_LayerData_Base
	{
		friend class FullyConnect_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		FullyConnect_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~FullyConnect_LayerData_CPU();


		//===========================
		// 初期化
		//===========================
	public:
		using FullyConnect_LayerData_Base::Initialize;

		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		ErrorCode Initialize(void);


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);

	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif