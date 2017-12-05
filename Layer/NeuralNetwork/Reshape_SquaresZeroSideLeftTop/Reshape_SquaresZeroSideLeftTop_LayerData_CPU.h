//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __RESHAPE_SQUARECENTERCROSS_LAYERDATA_CPU_H__
#define __RESHAPE_SQUARECENTERCROSS_LAYERDATA_CPU_H__

#include"Reshape_SquaresZeroSideLeftTop_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Reshape_SquaresZeroSideLeftTop_LayerData_CPU : public Reshape_SquaresZeroSideLeftTop_LayerData_Base
	{
		friend class Reshape_SquaresZeroSideLeftTop_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Reshape_SquaresZeroSideLeftTop_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Reshape_SquaresZeroSideLeftTop_LayerData_CPU();


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