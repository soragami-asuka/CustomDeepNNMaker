//=======================================
// �j���[�����l�b�g���[�N�{�̒�`
//=======================================
#ifndef __GRAVISBELL_I_NEURAL_NETWORK_H__
#define __GRAVISBELL_I_NEURAL_NETWORK_H__

#include"INNLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class INueralNetwork : public INNLayer
	{
	public:
		/** �R���X�g���N�^ */
		INueralNetwork(){}
		/** �f�X�g���N�^ */
		virtual ~INueralNetwork(){}

	public:
		/** ���C���[��ǉ�����.
			�ǉ��������C���[�̏��L����NeuralNetwork�Ɉڂ邽�߁A�������̊J�������Ȃǂ͑S��INeuralNetwork���ōs����.
			@param pLayer	�ǉ����郌�C���[�̃A�h���X. */
		virtual ErrorCode AddLayer(INNLayer* pLayer) = 0;
		/** ���C���[���폜����.
			@param guid	�폜���郌�C���[��GUID */
		virtual ErrorCode EraseLayer(const Gravisbell::GUID& guid) = 0;
		/** ���C���[��S�폜���� */
		virtual ErrorCode EraseAllLayer() = 0;

		/** �o�^����Ă��郌�C���[�����擾���� */
		virtual ErrorCode GetLayerCount()const = 0;
		/** ���C���[��GUID�w��Ŏ擾���� */
		virtual const INNLayer* GetLayerByGUID(const Gravisbell::GUID& guid) = 0;

		/** ���͐M���Ɋ��蓖�Ă��Ă���GUID���擾���� */
		virtual GUID GetInputGUID()const;
		/** �o�͐M���Ɋ��蓖�Ă�Ă��郌�C���[��GUID���擾���� */
		virtual GUID GetOutputLayerGUID()const;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif // __GRAVISBELL_I_NEURAL_NETWORK_H__
