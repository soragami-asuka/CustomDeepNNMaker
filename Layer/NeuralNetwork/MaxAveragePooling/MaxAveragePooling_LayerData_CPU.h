//======================================
// �v�[�����O���C���[�̃f�[�^
//======================================
#ifndef __MaxAveragePooling_LAYERDATA_CPU_H__
#define __MaxAveragePooling_LAYERDATA_CPU_H__

#include"MaxAveragePooling_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MaxAveragePooling_LayerData_CPU : public MaxAveragePooling_LayerData_Base
	{
		friend class MaxAveragePooling_CPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		MaxAveragePooling_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~MaxAveragePooling_LayerData_CPU();


		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif